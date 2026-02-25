/**
 * Exports all data needed for ML training to JSON files.
 *
 * Output directory: python-ml/training-data/
 *   users.json        — user profiles + tags + stats
 *   events.json       — events + real attendance stats
 *   spaces.json       — spaces + member/event stats
 *   interactions.json — positive interaction pairs
 *
 * Usage: npm run ml:export
 */

import "dotenv/config";
import fs from "fs";
import path from "path";
import postgres from "postgres";

const client = postgres(process.env.DATABASE_URL!);

const OUT_DIR = path.resolve(process.cwd(), "python-ml/training-data");

function write(filename: string, data: unknown) {
  const file = path.join(OUT_DIR, filename);
  fs.writeFileSync(file, JSON.stringify(data, null, 2));
  console.log(`  ✓ ${filename}  (${(data as unknown[]).length} records)`);
}

async function exportUsers() {
  // Profile fields + pre-computed stats
  const rows = await client`
    SELECT
      u.id::text,
      u.birthdate::text,
      u.gender,
      u.relationship_intent,
      u.smoking,
      u.drinking,
      u.activity_level,
      u.tags,
      (COUNT(DISTINCT ea.event_id) + COUNT(DISTINCT m.space_id))::int AS interaction_count
    FROM users u
    LEFT JOIN event_attendees ea
           ON ea.user_id = u.id AND ea.status = 'attended'
    LEFT JOIN members m
           ON m.user_id = u.id AND m.status = 'active'
    GROUP BY u.id, u.birthdate, u.gender, u.relationship_intent,
             u.smoking, u.drinking, u.activity_level, u.tags
    ORDER BY u.id
  `;

  return rows.map((u) => ({
    id: u.id,
    birthdate: u.birthdate ?? null,
    gender: u.gender ?? null,
    relationship_intent: u.relationship_intent ?? [],
    smoking: u.smoking ?? null,
    drinking: u.drinking ?? null,
    activity_level: u.activity_level ?? null,
    interaction_count: Number(u.interaction_count),
    tags: u.tags ?? [],
  }));
}

async function exportEvents() {
  /**
   * For each event:
   *   past     → attendee_count = users with status='attended' (real data)
   *   upcoming → attendee_count = users with status='going'    (intent)
   * avg_attendee_age follows the same logic.
   */
  const rows = await client`
    SELECT
      e.id::text,
      e.space_id::text,
      e.tags,
      e.starts_at::text,
      e.max_attendees,
      e.price,
      COUNT(ea_real.user_id)::int                            AS attended_count,
      AVG(date_part('year', age(u_real.birthdate)))::float   AS avg_age_attended,
      COUNT(ea_going.user_id)::int                           AS going_count,
      AVG(date_part('year', age(u_going.birthdate)))::float  AS avg_age_going
    FROM events e
    LEFT JOIN event_attendees ea_real
           ON ea_real.event_id = e.id AND ea_real.status = 'attended'
    LEFT JOIN users u_real
           ON u_real.id = ea_real.user_id AND u_real.birthdate IS NOT NULL
    LEFT JOIN event_attendees ea_going
           ON ea_going.event_id = e.id AND ea_going.status = 'going'
    LEFT JOIN users u_going
           ON u_going.id = ea_going.user_id AND u_going.birthdate IS NOT NULL
    GROUP BY e.id, e.space_id, e.tags, e.starts_at, e.max_attendees, e.price
    ORDER BY e.id
  `;

  return rows.map((e) => {
    const isCompleted = !!e.starts_at && new Date(e.starts_at).getTime() < Date.now();
    const attendeeCount = Number(isCompleted ? e.attended_count : e.going_count) || 0;
    const avgAge = isCompleted ? e.avg_age_attended : e.avg_age_going;

    return {
      id: e.id,
      space_id: e.space_id ?? null,
      tags: e.tags ?? [],
      starts_at: e.starts_at ?? null,
      max_attendees: e.max_attendees ? Number(e.max_attendees) : null,
      is_paid: e.price != null && Number(e.price) > 0,
      price_cents: e.price != null ? Number(e.price) : 0,
      attendee_count: attendeeCount,
      avg_attendee_age: avgAge != null ? Number(avgAge) : null,
    };
  });
}

async function exportSpaces() {
  const rows = await client`
    SELECT
      s.id::text,
      s.tags,
      COUNT(DISTINCT m.user_id)::int                        AS member_count,
      AVG(date_part('year', age(u.birthdate)))::float       AS avg_member_age,
      COUNT(DISTINCT e.id)::int                             AS event_count
    FROM spaces s
    LEFT JOIN members m ON m.space_id = s.id AND m.status = 'active'
    LEFT JOIN users u   ON u.id = m.user_id AND u.birthdate IS NOT NULL
    LEFT JOIN events e  ON e.space_id = s.id
              AND e.starts_at IS NOT NULL
    WHERE s.is_active = true
    GROUP BY s.id, s.tags
    ORDER BY s.id
  `;

  return rows.map((s) => ({
    id: s.id,
    tags: s.tags ?? [],
    member_count: Number(s.member_count),
    avg_member_age: s.avg_member_age != null ? Number(s.avg_member_age) : null,
    event_count: Number(s.event_count),
  }));
}

async function exportInteractions() {
  /**
   * Positive interactions with recency-weighted labels.
   *
   * weight = type_weight × recency_decay
   *   type_weight:  attended=1.0, going=0.7, member=0.9
   *   recency_decay: exp(-days_since / 180)  — half-life ≈ 6 months
   *
   * Final weight is clamped to [0.05, 1.0].
   */
  const HALF_LIFE_SECS = 180 * 86400; // 180 days in seconds
  const rows = await client`
    -- attended (past events) — strongest signal
    SELECT
      ea.user_id::text,
      ea.event_id::text                                                   AS item_id,
      'event'::text                                                       AS item_type,
      LEAST(1.0, GREATEST(0.05,
        1.0 * EXP(-EXTRACT(EPOCH FROM
          (NOW() - COALESCE(ea.attended_at, ea.registered_at))
        ) / ${HALF_LIFE_SECS})
      ))::float                                                           AS weight
    FROM event_attendees ea
    JOIN events e ON e.id = ea.event_id
    WHERE ea.status = 'attended' AND e.starts_at < NOW()

    UNION ALL

    -- going (upcoming events) — intent signal
    SELECT
      ea.user_id::text,
      ea.event_id::text                                                   AS item_id,
      'event'::text                                                       AS item_type,
      LEAST(1.0, GREATEST(0.05,
        0.7 * EXP(-EXTRACT(EPOCH FROM
          (NOW() - ea.registered_at)
        ) / ${HALF_LIFE_SECS})
      ))::float                                                           AS weight
    FROM event_attendees ea
    JOIN events e ON e.id = ea.event_id
    WHERE ea.status = 'going' AND e.starts_at >= NOW()

    UNION ALL

    -- space members
    SELECT
      m.user_id::text,
      m.space_id::text                                                    AS item_id,
      'space'::text                                                       AS item_type,
      LEAST(1.0, GREATEST(0.05,
        0.9 * EXP(-EXTRACT(EPOCH FROM
          (NOW() - m.joined_at)
        ) / ${HALF_LIFE_SECS})
      ))::float                                                           AS weight
    FROM members m
    WHERE m.status = 'active'

  `;

  return rows.map((r) => ({
    user_id: r.user_id,
    item_id: r.item_id,
    item_type: r.item_type,
    weight: Number(r.weight),
  }));
}

async function main() {
  console.log("Exporting ML training data...\n");

  fs.mkdirSync(OUT_DIR, { recursive: true });

  const [users, events, spaces, interactions] = await Promise.all([
    exportUsers(),
    exportEvents(),
    exportSpaces(),
    exportInteractions(),
  ]);

  write("users.json", users);
  write("events.json", events);
  write("spaces.json", spaces);
  write("interactions.json", interactions);

  console.log(`\nOutput: ${OUT_DIR}`);
  await client.end();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
