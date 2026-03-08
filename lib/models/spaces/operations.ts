/**
 * Spaces Operations
 */

import { db } from "@/lib/db/drizzle";
import { spaces, type Space } from "./schema";
import { members } from "@/lib/models/members/schema";
import { events } from "@/lib/models/events/schema";
import { embeddings } from "@/lib/models/embeddings/schema";
import { eq, and, ne, sql, desc, gte, inArray } from "drizzle-orm";
import { embedSpace } from "@/lib/ml/client";
import { recordImpression } from "@/lib/models/impressions/operations";
import type { CreateSpaceInput, UpdateSpaceInput } from "./validator";


// ============================================
// SPACE CRUD OPERATIONS
// ============================================

export interface CreateSpaceResult {
  space: Space;
}

export async function createSpace(params: CreateSpaceInput & { ownerId: string }): Promise<CreateSpaceResult> {
  const result = await db.transaction(async (tx) => {
    const [space] = await tx
      .insert(spaces)
      .values({
        name: params.name,
        slug: params.slug,
        description: params.description,
        cover: params.cover,
        images: params.images ?? [],
        visibility: params.visibility || "public",
        joinPolicy: params.joinPolicy || "open",
        categories: params.categories ?? [],
        ownerId: params.ownerId,
      })
      .returning();

    await tx.insert(members).values({
      spaceId: space.id,
      userId: params.ownerId,
      role: "owner",
      status: "active",
    });

    return { space };
  });

  // Generate and store embedding
  const { space } = result;
  const embedding = await embedSpace({
    categories: space.categories ?? [],
    memberCount: 1, // owner is the first member
    eventCount: 0,
  });
  if (embedding) {
    await db
      .insert(embeddings)
      .values({ entityId: space.id, entityType: "space", embedding })
      .onConflictDoUpdate({
        target: [embeddings.entityId, embeddings.entityType],
        set: { embedding, updatedAt: new Date() },
      });
  }

  return result;
}

export async function getSpaceById(id: string, userId?: string): Promise<Space | null> {
  const result = await db.query.spaces.findFirst({
    where: eq(spaces.id, id),
  });
  if (!result) return null;

  // Track the visit server-side (fire-and-forget)
  if (userId) recordImpression(userId, id, "space", "viewed");

  return result;
}

export async function getSpaceBySlug(slug: string): Promise<Space | null> {
  const result = await db.query.spaces.findFirst({
    where: eq(spaces.slug, slug),
  });
  return result || null;
}

export async function getAllSpaces(): Promise<Space[]> {
  return db.query.spaces.findMany({
    orderBy: (spaces, { desc }) => [desc(spaces.createdAt)],
  });
}

/**
 * Search spaces by tags. matchAll=true requires all tags, false requires at least one.
 */
export async function getSpacesByCategories(
  categories: string[],
  matchAll: boolean = false,
): Promise<Space[]> {
  if (categories.length === 0) return [];

  const catArray = `{${categories.join(",")}}`;
  const operator = matchAll ? "@>" : "&&";

  return await db
    .select()
    .from(spaces)
    .where(
      and(
        sql`${spaces.categories} ${sql.raw(operator)} ${catArray}::text[]`,
        eq(spaces.visibility, "public"),
      ),
    )
    .orderBy(desc(spaces.createdAt));
}

/**
 * Update space
 */
export async function updateSpace(id: string, data: UpdateSpaceInput): Promise<Space | null> {
  const [updated] = await db
    .update(spaces)
    .set({ ...data, updatedAt: new Date() })
    .where(eq(spaces.id, id))
    .returning();

  if (updated && (data.name !== undefined || data.description !== undefined || data.categories !== undefined)) {
    const embedding = await embedSpace({
      categories: updated.categories ?? [],
      memberCount: 0,
      eventCount: 0,
    });
    if (embedding) {
      await db
        .insert(embeddings)
        .values({ entityId: updated.id, entityType: "space", embedding })
        .onConflictDoUpdate({
          target: [embeddings.entityId, embeddings.entityType],
          set: { embedding, updatedAt: new Date() },
        });
    }
  }

  return updated || null;
}

/**
 * AI-recommended events for a space — finds upcoming events whose embeddings
 * are closest to this space's embedding (cross-entity cosine similarity).
 * Excludes events that already belong to this space.
 */
export async function getSpaceRecommendedEvents(
  spaceId: string,
  limit = 6,
) {
  const row = await db.query.embeddings.findFirst({
    where: and(eq(embeddings.entityId, spaceId), eq(embeddings.entityType, "space")),
    columns: { embedding: true },
  });
  if (!row) return [];

  const vec = `[${row.embedding.join(",")}]`;
  const rows = await db
    .select({ id: events.id })
    .from(embeddings)
    .innerJoin(events, eq(sql`${events.id}::text`, embeddings.entityId))
    .where(
      and(
        eq(embeddings.entityType, "event"),
        ne(events.spaceId, spaceId),
        gte(events.startsAt, new Date()),
      ),
    )
    .orderBy(sql`${embeddings.embedding} <=> ${sql.raw(`'${vec}'::vector`)}`)
    .limit(limit);

  if (!rows.length) return [];

  const ids = rows.map((r) => r.id);
  return db.query.events.findMany({
    where: (e, { inArray }) => inArray(e.id, ids),
  });
}

/**
 * Delete space
 */
export async function deleteSpace(id: string): Promise<boolean> {
  const result = await db.delete(spaces).where(eq(spaces.id, id)).returning();
  return result.length > 0;
}
