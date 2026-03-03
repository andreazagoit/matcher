import { db } from "@/lib/db/drizzle";
import { categories } from "./schema";
import { embeddings } from "../embeddings/schema";
import { asc, eq } from "drizzle-orm";

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

  let mlVector: number[] = [];
  let graphVector: number[] = [];
  try {
    const mlResponse = await fetch(
      `${process.env.ML_SERVICE_URL ?? "http://localhost:8000"}/embed/category`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      },
    );
    if (!mlResponse.ok)
      throw new Error(`ML service returned ${mlResponse.status}`);
    const mlData = await mlResponse.json();
    mlVector = mlData.category_embedding;
    graphVector = mlData.embedding;
  } catch (err) {
    console.warn("ML service unavailable for category embedding:", err);
    mlVector = new Array(64).fill(0.0);
    graphVector = new Array(256).fill(0.0);
  }

  await db.transaction(async (tx) => {
    await tx
      .insert(categories)
      .values({ id, name: id, embedding: mlVector });
    await tx
      .insert(embeddings)
      .values({ entityId: id, entityType: "category", embedding: graphVector });
  });

  return id;
}
