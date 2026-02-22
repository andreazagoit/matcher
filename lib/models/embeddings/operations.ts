/**
 * Embeddings operations — generate and persist entity embeddings.
 *
 * Currently powered by OpenAI text-embedding-3-small (1536-dim).
 * All entity types share the same embeddings table, enabling cross-entity
 * similarity search via pgvector.
 *
 * To switch to an ML model in the future: replace the `generate*Text`
 * helpers with structured feature vectors and swap `generateEmbedding`
 * for a call to the ML service. The public API and DB logic stay identical.
 */

import { db } from "@/lib/db/drizzle";
import { generateEmbedding } from "@/lib/embeddings";
import { embeddings } from "./schema";
import { eq, and } from "drizzle-orm";

// ─── Input types ───────────────────────────────────────────────────────────────

export interface UserEmbedInput {
  tags: { tag: string; weight: number }[];
  birthdate?: string | null; // "YYYY-MM-DD"
  gender?: string | null;
  relationshipIntent?: string[] | null;
  jobTitle?: string | null;
  educationLevel?: string | null;
  smoking?: string | null;
  drinking?: string | null;
  activityLevel?: string | null;
  religion?: string | null;
}

export interface EventEmbedInput {
  title: string;
  description?: string | null;
  tags?: string[];
}

export interface SpaceEmbedInput {
  name: string;
  description?: string | null;
  tags?: string[];
}

// ─── Text builders ─────────────────────────────────────────────────────────────

function buildUserText({
  tags, birthdate, gender, relationshipIntent,
  jobTitle, educationLevel, smoking, drinking, activityLevel, religion,
}: UserEmbedInput): string {
  const parts: string[] = [];

  if (birthdate) {
    const age = new Date().getFullYear() - new Date(birthdate).getFullYear();
    parts.push(`Age: ${age}`);
  }
  if (gender)              parts.push(`Gender: ${gender}`);
  if (jobTitle)            parts.push(`Job: ${jobTitle}`);
  if (educationLevel)      parts.push(`Education: ${educationLevel}`);
  if (relationshipIntent?.length) parts.push(`Looking for: ${relationshipIntent.join(", ")}`);
  if (smoking)             parts.push(`Smoking: ${smoking}`);
  if (drinking)            parts.push(`Drinking: ${drinking}`);
  if (activityLevel)       parts.push(`Activity: ${activityLevel}`);
  if (religion)            parts.push(`Religion: ${religion}`);

  // Weighted interests: high-weight tags repeated to increase their influence
  // weight >= 0.8 → mentioned 3×, >= 0.5 → 2×, < 0.5 → 1×
  if (tags.length > 0) {
    const weighted = tags.flatMap(({ tag, weight }) => {
      const reps = weight >= 0.8 ? 3 : weight >= 0.5 ? 2 : 1;
      return Array(reps).fill(tag);
    });
    parts.push(`Interests: ${weighted.join(", ")}`);
  }

  return parts.join(". ");
}

function buildEventText({ title, description, tags }: EventEmbedInput): string {
  const parts = [title];
  if (description) parts.push(description);
  if (tags?.length) parts.push(`Tags: ${tags.join(", ")}`);
  return parts.join("\n");
}

function buildSpaceText({ name, description, tags }: SpaceEmbedInput): string {
  const parts = [name];
  if (description) parts.push(description);
  if (tags?.length) parts.push(`Tags: ${tags.join(", ")}`);
  return parts.join("\n");
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
 * Call after creating a user or updating their interests.
 */
export async function embedUser(
  entityId: string,
  data: UserEmbedInput,
): Promise<void> {
  const text = buildUserText(data);
  if (!text.trim()) return;
  const embedding = await generateEmbedding(text);
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
  const text = buildEventText(data);
  if (!text.trim()) return;
  const embedding = await generateEmbedding(text);
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
  const text = buildSpaceText(data);
  if (!text.trim()) return;
  const embedding = await generateEmbedding(text);
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
