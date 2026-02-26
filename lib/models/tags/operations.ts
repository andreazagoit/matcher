import { db } from "@/lib/db/drizzle";
import { tags } from "./schema";
import { embeddings } from "../embeddings/schema";
import { asc, eq } from "drizzle-orm";

/**
 * Get all tag categories and their associated tags from PostgreSQL.
 */
export async function getTagCategories() {
  const dbTags = await db.select().from(tags).orderBy(asc(tags.name));

  const categoriesMap: Record<string, string[]> = {};
  for (const tag of dbTags) {
    if (!categoriesMap[tag.category]) {
      categoriesMap[tag.category] = [];
    }
    categoriesMap[tag.category].push(tag.name);
  }

  return Object.entries(categoriesMap).map(([category, tagList]) => ({
    category,
    tags: tagList,
  }));
}

/**
 * Get all valid tags as a flat list from PostgreSQL.
 */
export async function getAllTags() {
  const dbTags = await db.select({ name: tags.name }).from(tags);
  return dbTags.map((t) => t.name);
}

/**
 * Validate a list of tags.
 */
export async function validateTags(tagsInput: string[]): Promise<string | null> {
  const validTags = new Set(await getAllTags());
  return tagsInput.find((t) => !validTags.has(t)) || null;
}

/**
 * Create a new tag. Defers embeddings generation (both 64d and 256d) 
 * to the Python ML Service and saves the result in Postgres.
 */
export async function createTag(name: string, category: string) {
  const sanitizedName = name.toLowerCase().replace(/[^a-z0-9_]/g, "_");

  // Fast path: if tag already exists in the database, skip ML pipeline
  const existing = await db.select({ id: tags.id }).from(tags).where(eq(tags.id, sanitizedName)).limit(1);
  if (existing.length > 0) {
    return sanitizedName;
  }

  let mlVector: number[] = [];
  let graphVector: number[] = [];

  try {
    const mlResponse = await fetch("http://localhost:8000/embed/tag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name }),
    });

    if (!mlResponse.ok) {
      throw new Error(`Python ML Service returned status ${mlResponse.status}`);
    }

    const mlData = await mlResponse.json();
    mlVector = mlData.tag_embedding; // 64 dimensions from OpenAI
    graphVector = mlData.embedding;  // 256 dimensions from HGT graph projector

  } catch (error) {
    console.warn("Failed to reach Python ML Service for tag generation.", error);
    mlVector = new Array(64).fill(0.0);
    graphVector = new Array(256).fill(0.0);
  }

  await db.transaction(async (tx) => {
    // Insert into 'tags' table with 64d
    await tx.insert(tags).values({
      id: sanitizedName,
      name: sanitizedName,
      category,
      embedding: mlVector,
    });

    // Insert into 'embeddings' table with 256d
    await tx.insert(embeddings).values({
      entityId: sanitizedName,
      entityType: "tag",
      embedding: graphVector,
    });
  });

  return sanitizedName;
}
