/**
 * Spaces Operations
 */

import { db } from "@/lib/db/drizzle";
import { spaces, type Space } from "./schema";
import { members } from "@/lib/models/members/schema";
import { eq, and, sql, desc } from "drizzle-orm";
import { generateEmbedding } from "@/lib/embeddings";
import { recordImpression } from "@/lib/models/impressions/operations";

export function generateSlug(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

// ============================================
// SPACE CRUD OPERATIONS
// ============================================

export interface CreateSpaceResult {
  space: Space;
}

export async function createSpace(params: {
  name: string;
  slug?: string;
  description?: string;
  ownerId: string;
  visibility?: "public" | "private" | "hidden";
  joinPolicy?: "open" | "apply" | "invite_only";
  image?: string;
  categories?: string[];
}): Promise<CreateSpaceResult> {
  const slug = params.slug || generateSlug(params.name);

  const result = await db.transaction(async (tx) => {
    const [space] = await tx
      .insert(spaces)
      .values({
        name: params.name,
        slug,
        description: params.description,
        image: params.image,
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

  // Generate embedding in background
  const { space } = result;
  const embText = [space.name, space.description, space.categories?.length ? `Categories: ${space.categories.join(", ")}` : ""].filter(Boolean).join("\n");
  if (embText.trim()) {
    generateEmbedding(embText)
      .then((embedding) => db.update(spaces).set({ embedding }).where(eq(spaces.id, space.id)))
      .catch(() => {});
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
        eq(spaces.isActive, true),
      ),
    )
    .orderBy(desc(spaces.createdAt));
}

/**
 * Update space
 */
export async function updateSpace(
  id: string,
  data: Partial<Pick<Space, "name" | "slug" | "description" | "isActive" | "visibility" | "image" | "joinPolicy" | "categories">>
): Promise<Space | null> {
  const [updated] = await db
    .update(spaces)
    .set({ ...data, updatedAt: new Date() })
    .where(eq(spaces.id, id))
    .returning();

  if (updated && (data.name !== undefined || data.description !== undefined || data.categories !== undefined)) {
    const embText = [updated.name, updated.description, updated.categories?.length ? `Categories: ${updated.categories.join(", ")}` : ""].filter(Boolean).join("\n");
    if (embText.trim()) {
      generateEmbedding(embText)
        .then((embedding) => db.update(spaces).set({ embedding }).where(eq(spaces.id, updated.id)))
        .catch(() => {});
    }
  }

  return updated || null;
}

/**
 * Delete space
 */
export async function deleteSpace(id: string): Promise<boolean> {
  const result = await db.delete(spaces).where(eq(spaces.id, id)).returning();
  return result.length > 0;
}
