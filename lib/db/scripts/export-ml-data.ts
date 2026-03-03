/**
 * Exports all data needed for ML training to JSON files.
 *
 * Output directory: python-ml/training-data/
 *   users.json               — user profiles + stats (no categories array)
 *   events.json              — events + real attendance stats + categories
 *   spaces.json              — spaces + member/event stats + categories
 *   categories.json          — category nodes with 64d embeddings
 *   category_impressions.json— user→category edges (liked + viewed impressions)
 *   event_attendees.json     — user→event weighted edges
 *   members.json             — user→space weighted edges
 *   connections.json         — user→user connection edges
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
  const rows = await client`
    SELECT
      u.id::text,
      u.birthdate::text,
      u.gender,
      u.relationship_intent,
      u.smoking,
      u.drinking,
      u.activity_level
    FROM users u
    ORDER BY u.id
  `;

  return rows.map((u) => ({
    id: u.id,
    birthdate: u.birthdate ?? null,
    gender: u.gender ?? null,
    relationshipIntent: u.relationship_intent ?? [],
    smoking: u.smoking ?? null,
    drinking: u.drinking ?? null,
    activityLevel: u.activity_level ?? null,
  }));
}

async function exportEvents() {
  const rows = await client`
    SELECT
      e.id::text,
      e.space_id::text,
      e.categories,
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
    GROUP BY e.id, e.space_id, e.categories, e.starts_at, e.max_attendees, e.price
    ORDER BY e.id
  `;

  return rows.map((e) => {
    const isCompleted = !!e.starts_at && new Date(e.starts_at).getTime() < Date.now();
    const attendeeCount = Number(isCompleted ? e.attended_count : e.going_count) || 0;
    const avgAge = isCompleted ? e.avg_age_attended : e.avg_age_going;

    return {
      id: e.id,
      spaceId: e.space_id ?? null,
      categories: e.categories ?? [],
      startsAt: e.starts_at ?? null,
      maxAttendees: e.max_attendees ? Number(e.max_attendees) : null,
      isPaid: e.price != null && Number(e.price) > 0,
      priceCents: e.price != null ? Number(e.price) : 0,
      attendeeCount: attendeeCount,
      avgAttendeeAge: avgAge != null ? Number(avgAge) : null,
    };
  });
}

async function exportCategories() {
  const rows = await client`
    SELECT
      id,
      name,
      embedding::text
    FROM categories
    ORDER BY id ASC
  `;
  return rows.map((c) => ({
    id: c.id,
    name: c.name,
    embedding: c.embedding ? JSON.parse(c.embedding) : null,
  }));
}

/**
 * Export user→category edges from impressions.
 * Weight: liked=1.0, viewed=0.3 (page visit is weaker signal than explicit like).
 * Multiple impressions of the same pair are summed and capped at 1.0.
 */
async function exportCategoryImpressions() {
  const HALF_LIFE_SECS = 90 * 86400; // 90 days — category interest decays faster

  const rows = await client`
    SELECT
      user_id::text,
      item_id         AS category_id,
      action,
      LEAST(1.0, GREATEST(0.05,
        CASE action
          WHEN 'liked'  THEN 1.0
          WHEN 'viewed' THEN 0.3
          ELSE 0.1
        END
        * EXP(-EXTRACT(EPOCH FROM (NOW() - created_at)) / ${HALF_LIFE_SECS})
      ))::float AS weight,
      created_at::text
    FROM impressions
    WHERE item_type = 'category'
      AND action IN ('liked', 'viewed')
    ORDER BY user_id, item_id, created_at ASC
  `;

  // Aggregate multiple impressions of the same user→category pair
  const aggregated = new Map<string, { userId: string; itemId: string; weight: number; created_at: string }>();
  for (const r of rows) {
    const key = `${r.user_id}:${r.category_id}`;
    const existing = aggregated.get(key);
    if (existing) {
      existing.weight = Math.min(1.0, existing.weight + Number(r.weight));
    } else {
      aggregated.set(key, {
        userId: r.user_id,
        itemId: r.category_id,
        weight: Number(r.weight),
        created_at: r.created_at,
      });
    }
  }

  return Array.from(aggregated.values());
}

async function exportSpaces() {
  const rows = await client`
    SELECT
      s.id::text,
      s.categories,
      COUNT(DISTINCT m.user_id)::int                        AS member_count,
      AVG(date_part('year', age(u.birthdate)))::float       AS avg_member_age,
      COUNT(DISTINCT e.id)::int                             AS event_count
    FROM spaces s
    LEFT JOIN members m ON m.space_id = s.id AND m.status = 'active'
    LEFT JOIN users u   ON u.id = m.user_id AND u.birthdate IS NOT NULL
    LEFT JOIN events e  ON e.space_id = s.id
              AND e.starts_at IS NOT NULL
    WHERE s.is_active = true
    GROUP BY s.id, s.categories
    ORDER BY s.id
  `;

  return rows.map((s) => ({
    id: s.id,
    categories: s.categories ?? [],
    memberCount: Number(s.member_count),
    avgMemberAge: s.avg_member_age != null ? Number(s.avg_member_age) : null,
    eventCount: Number(s.event_count),
  }));
}

async function exportEventAttendees() {
  const HALF_LIFE_SECS = 180 * 86400;
  const rows = await client`
    SELECT
      ea.user_id::text,
      ea.event_id::text,
      LEAST(1.0, GREATEST(0.05,
        1.0 * EXP(-EXTRACT(EPOCH FROM
          (NOW() - COALESCE(ea.attended_at, ea.registered_at))
        ) / ${HALF_LIFE_SECS})
      ))::float AS weight
    FROM event_attendees ea
    JOIN events e ON e.id = ea.event_id
    WHERE ea.status = 'attended' AND e.starts_at < NOW()

    UNION ALL

    SELECT
      ea.user_id::text,
      ea.event_id::text,
      LEAST(1.0, GREATEST(0.05,
        0.7 * EXP(-EXTRACT(EPOCH FROM
          (NOW() - ea.registered_at)
        ) / ${HALF_LIFE_SECS})
      ))::float AS weight
    FROM event_attendees ea
    JOIN events e ON e.id = ea.event_id
    WHERE ea.status = 'going' AND e.starts_at >= NOW()
  `;

  return rows.map((r) => ({
    userId: r.user_id,
    eventId: r.event_id,
    weight: Number(r.weight),
  }));
}

async function exportMembers() {
  const HALF_LIFE_SECS = 180 * 86400;
  const rows = await client`
    SELECT
      m.user_id::text,
      m.space_id::text,
      LEAST(1.0, GREATEST(0.05,
        0.9 * EXP(-EXTRACT(EPOCH FROM
          (NOW() - m.joined_at)
        ) / ${HALF_LIFE_SECS})
      ))::float AS weight
    FROM members m
    WHERE m.status = 'active'
  `;

  return rows.map((r) => ({
    userId: r.user_id,
    spaceId: r.space_id,
    weight: Number(r.weight),
  }));
}

async function exportConnections() {
  const rows = await client`
    SELECT
      initiator_id::text,
      recipient_id::text,
      status::text,
      created_at::text
    FROM connections
    ORDER BY created_at ASC
  `;
  return rows.map((c) => ({
    initiator_id: c.initiator_id,
    recipient_id: c.recipient_id,
    status: c.status,
    created_at: c.created_at,
  }));
}

async function main() {
  console.log("Exporting ML training data...\n");

  fs.mkdirSync(OUT_DIR, { recursive: true });

  const [users, events, spaces, categories, categoryImpressions, eventAttendees, members, connections] =
    await Promise.all([
      exportUsers(),
      exportEvents(),
      exportSpaces(),
      exportCategories(),
      exportCategoryImpressions(),
      exportEventAttendees(),
      exportMembers(),
      exportConnections(),
    ]);

  write("users.json", users);
  write("events.json", events);
  write("spaces.json", spaces);
  write("categories.json", categories);
  write("category_impressions.json", categoryImpressions);
  write("event_attendees.json", eventAttendees);
  write("members.json", members);
  write("connections.json", connections);

  console.log(`\nOutput: ${OUT_DIR}`);
  await client.end();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
