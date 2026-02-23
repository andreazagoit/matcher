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
import { eq, and } from "drizzle-orm";

const ML_SERVICE_URL =
  process.env.ML_SERVICE_URL ?? "http://localhost:8000";

// ─── Input types ───────────────────────────────────────────────────────────────

export interface UserEmbedInput {
  tags: { tag: string; weight: number }[];
  birthdate?: string | null;
  gender?: string | null;
  relationshipIntent?: string[] | null;
  smoking?: string | null;
  drinking?: string | null;
  activityLevel?: string | null;
  interactionCount?: number;
  conversationCount?: number;
  // kept for forward compatibility; not used by the current model
  jobTitle?: string | null;
  educationLevel?: string | null;
  religion?: string | null;
}

export interface EventEmbedInput {
  tags?: string[];
  startsAt?: string | null;        // ISO string — used to compute daysUntilEvent
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
  const tagWeights: Record<string, number> = {};
  for (const { tag, weight } of data.tags) {
    tagWeights[tag] = weight;
  }

  const embedding = await callMlEmbed("user", {
    tag_weights:          tagWeights,
    birthdate:            data.birthdate ?? null,
    gender:               data.gender ?? null,
    relationship_intent:  data.relationshipIntent ?? [],
    smoking:              data.smoking ?? null,
    drinking:             data.drinking ?? null,
    activity_level:       data.activityLevel ?? null,
    interaction_count:    data.interactionCount ?? 0,
    conversation_count:   data.conversationCount ?? 0,
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
  entityType: "user" | "event" | "space",
): Promise<number[] | null> {
  const row = await db.query.embeddings.findFirst({
    where: and(
      eq(embeddings.entityId, entityId),
      eq(embeddings.entityType, entityType),
    ),
  });
  return row?.embedding ?? null;
}
