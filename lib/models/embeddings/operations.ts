
/**
 * Embeddings operations — generate and persist entity embeddings.
 *
 * Powered by the HGT-lite ML service (64-dim behavioural embeddings).
 * The service runs at ML_SERVICE_URL (default: http://localhost:8000).
 *
 * Public API is identical to the previous OpenAI version — only the
 * internal implementation changed. Callers need no updates.
 */

import { db } from "@/lib/db/drizzle";
import { embeddings } from "./schema";
import { eq, and, sql } from "drizzle-orm";

const ML_SERVICE_URL =
  process.env.ML_SERVICE_URL ?? "http://localhost:8000";

// ─── Input types ───────────────────────────────────────────────────────────────

export interface UserEmbedInput {
  tags: string[];
  birthdate?: string | null;
  gender?: string | null;
  relationshipIntent?: string[] | null;
  smoking?: string | null;
  drinking?: string | null;
  activityLevel?: string | null;
  interactionCount?: number;
}

export interface EventEmbedInput {
  tags?: string[];
  startsAt?: string | null;        // ISO string — used to compute daysUntilEvent
  priceCents?: number | null;
  avgAttendeeAge?: number | null;
  attendeeCount?: number;
  maxAttendees?: number | null;
  isPaid?: boolean;
  // kept for forward compatibility; not used by the current model
  title?: string;
  description?: string | null;
}

export interface SpaceEmbedInput {
  tags?: string[];
  memberCount?: number;
  avgMemberAge?: number | null;
  eventCount?: number;
  // kept for forward compatibility; not used by the current model
  name?: string;
  description?: string | null;
}

// ─── ML service call ───────────────────────────────────────────────────────────

async function callMlEmbed(
  entityType: "user" | "event" | "space",
  payload: Record<string, unknown>,
): Promise<number[]> {
  const res = await fetch(`${ML_SERVICE_URL}/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ entity_type: entityType, [entityType]: payload }),
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(
      `ML service error ${res.status} for ${entityType}: ${body}`,
    );
  }

  const json = await res.json();
  return json.embedding as number[];
}

// ─── DB persistence ────────────────────────────────────────────────────────────

async function upsertEmbedding(
  entityId: string,
  entityType: "user" | "event" | "space",
  embedding: number[],
): Promise<void> {
  await db
    .insert(embeddings)
    .values({ entityId, entityType, embedding, updatedAt: new Date() })
    .onConflictDoUpdate({
      target: [embeddings.entityId, embeddings.entityType],
      set: { embedding, updatedAt: new Date() },
    });
}

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Generate and save embedding for a user.
 * Call after creating a user or updating their profile / interests.
 */
export async function embedUser(
  entityId: string,
  data: UserEmbedInput,
): Promise<void> {
  const embedding = await callMlEmbed("user", {
    tags:                 data.tags,
    birthdate:            data.birthdate ?? null,
    gender:               data.gender ?? null,
    relationship_intent:  data.relationshipIntent ?? [],
    smoking:              data.smoking ?? null,
    drinking:             data.drinking ?? null,
    activity_level:       data.activityLevel ?? null,
    interaction_count:    data.interactionCount ?? 0,
  });

  await upsertEmbedding(entityId, "user", embedding);
}

/**
 * Generate and save embedding for an event.
 * Call after creating or updating an event.
 */
export async function embedEvent(
  entityId: string,
  data: EventEmbedInput,
): Promise<void> {
  const daysUntilEvent = data.startsAt
    ? Math.round((new Date(data.startsAt).getTime() - Date.now()) / 86_400_000)
    : null;

  const embedding = await callMlEmbed("event", {
    tags:               data.tags ?? [],
    starts_at:          data.startsAt ?? null,
    price_cents:        data.priceCents ?? null,
    avg_attendee_age:   data.avgAttendeeAge ?? null,
    attendee_count:     data.attendeeCount ?? 0,
    days_until_event:   daysUntilEvent,
    max_attendees:      data.maxAttendees ?? null,
    is_paid:            data.isPaid ?? false,
  });

  await upsertEmbedding(entityId, "event", embedding);
}

/**
 * Generate and save embedding for a space.
 * Call after creating or updating a space.
 */
export async function embedSpace(
  entityId: string,
  data: SpaceEmbedInput,
): Promise<void> {
  const embedding = await callMlEmbed("space", {
    tags:            data.tags ?? [],
    member_count:    data.memberCount ?? 0,
    avg_member_age:  data.avgMemberAge ?? null,
    event_count:     data.eventCount ?? 0,
  });

  await upsertEmbedding(entityId, "space", embedding);
}

/**
 * Get the stored embedding for an entity.
 * Returns null if no embedding has been generated yet.
 */
export async function getStoredEmbedding(
  entityId: string,
  entityType: "user" | "event" | "space" | "tag",
): Promise<number[] | null> {
  const row = await db.query.embeddings.findFirst({
    where: and(
      eq(embeddings.entityId, entityId),
      eq(embeddings.entityType, entityType),
    ),
  });
  return row?.embedding ?? null;
}

// ─── Internal helper ───────────────────────────────────────────────────────────

async function _getUserVector(userId: string): Promise<string | null> {
  const row = await db.query.embeddings.findFirst({
    where: and(
      eq(embeddings.entityId, userId),
      eq(embeddings.entityType, "user"),
    ),
    columns: { embedding: true },
  });
  return row ? `[${row.embedding.join(",")}]` : null;
}

// ─── Public recommendation queries ────────────────────────────────────────────

/**
 * Users with the most similar embeddings (excludes self).
 */
export async function recommendUsersForUser(
  userId: string,
  limit = 10,
  offset = 0,
): Promise<string[]> {
  const vec = await _getUserVector(userId);
  if (!vec) return [];

  const rows = await db.execute<{ entity_id: string }>(sql`
    SELECT entity_id
    FROM   embeddings
    WHERE  entity_type = 'user'
      AND  entity_id   != ${userId}
    ORDER BY embedding <=> ${sql.raw(`'${vec}'::vector`)}
    LIMIT  ${sql.raw(String(limit))}
    OFFSET ${sql.raw(String(offset))}
  `);

  return rows.map((r) => r.entity_id);
}

/**
 * Events closest to the user's embedding (by entity_id, caller applies business filters).
 */
export async function recommendEventsForUser(
  userId: string,
  limit = 10,
  offset = 0,
  excludeIds: string[] = [],
): Promise<string[]> {
  const vec = await _getUserVector(userId);
  if (!vec) return [];

  const excludeSet  = new Set(excludeIds);
  const fetchExtra  = limit + excludeIds.length;
  const rows = await db.execute<{ entity_id: string }>(sql`
    SELECT entity_id
    FROM   embeddings
    WHERE  entity_type = 'event'
    ORDER BY embedding <=> ${sql.raw(`'${vec}'::vector`)}
    LIMIT  ${sql.raw(String(fetchExtra))}
    OFFSET ${sql.raw(String(offset))}
  `);

  return rows
    .map((r) => r.entity_id)
    .filter((id) => !excludeSet.has(id))
    .slice(0, limit);
}

/**
 * Spaces closest to the user's embedding (by entity_id, caller applies business filters).
 */
export async function recommendSpacesForUser(
  userId: string,
  limit = 10,
  offset = 0,
  excludeIds: string[] = [],
): Promise<string[]> {
  const vec = await _getUserVector(userId);
  if (!vec) return [];

  const excludeSet  = new Set(excludeIds);
  const fetchExtra  = limit + excludeIds.length;
  const rows = await db.execute<{ entity_id: string }>(sql`
    SELECT entity_id
    FROM   embeddings
    WHERE  entity_type = 'space'
    ORDER BY embedding <=> ${sql.raw(`'${vec}'::vector`)}
    LIMIT  ${sql.raw(String(fetchExtra))}
    OFFSET ${sql.raw(String(offset))}
  `);

  return rows
    .map((r) => r.entity_id)
    .filter((id) => !excludeSet.has(id))
    .slice(0, limit);
}
