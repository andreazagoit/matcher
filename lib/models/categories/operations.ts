import { db } from "@/lib/db/drizzle";
import { categories } from "./schema";
import { embeddings } from "../embeddings/schema";
import { asc, eq } from "drizzle-orm";
import { embedCategory } from "@/lib/ml/client";

export async function getCategories() {
  return db.select().from(categories).orderBy(asc(categories.name));
}

export async function getCategoryById(id: string) {
  const result = await db
    .select()
    .from(categories)
    .where(eq(categories.id, id))
    .limit(1);
  return result[0] ?? null;
}

export async function getAllCategoryNames(): Promise<string[]> {
  const rows = await db.select({ name: categories.name }).from(categories);
  return rows.map((r) => r.name);
}

export async function validateCategories(input: string[]): Promise<string | null> {
  const valid = new Set(await getAllCategoryNames());
  return input.find((c) => !valid.has(c)) ?? null;
}

export async function createCategory(name: string): Promise<string> {
  const id = name.toLowerCase().replace(/[^a-z0-9_]/g, "_");
  const existing = await db
    .select({ id: categories.id })
    .from(categories)
    .where(eq(categories.id, id))
    .limit(1);
  if (existing.length > 0) return id;

  const emb = await embedCategory(name);
  const mlVector = emb?.categoryEmbedding ?? new Array(64).fill(0.0);
  const graphVector = emb?.embedding ?? new Array(256).fill(0.0);

  await db.transaction(async (tx) => {
    await tx
      .insert(categories)
      .values({ id, name: id, embedding: mlVector });
    await tx
      .insert(embeddings)
      .values({ entityId: id, entityType: "category", embedding: graphVector })
      .onConflictDoUpdate({
        target: [embeddings.entityId, embeddings.entityType],
        set: { embedding: graphVector, updatedAt: new Date() },
      });
  });

  return id;
}
