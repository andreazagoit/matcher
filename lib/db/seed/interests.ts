import { db } from "../drizzle";
import { impressions } from "../../models/impressions/schema";
import { CATEGORIES } from "../../models/categories/data";
import { createCategory } from "../../models/categories/operations";

function pickRandom<T>(arr: T[], count: number): T[] {
  return [...arr].sort(() => Math.random() - 0.5).slice(0, count);
}

/**
 * Seed all categories into the `categories` table and generate their embeddings.
 * Safe to call multiple times — createCategory skips existing ones.
 */
export async function seedCategories() {
  console.log(`\n🏷️  Seeding ${CATEGORIES.length} categories...`);
  let created = 0;

  for (const name of CATEGORIES) {
    await createCategory(name);
    console.log(`  ✓ ${name}`);
    created++;
  }

  console.log(`  → ${created} categories seeded`);
}

/**
 * Seed category interests for a list of user IDs.
 * Each user gets 3–6 random liked category impressions.
 */
export async function seedProfiles(userIds: string[]) {
  console.log(`\n📋 Seeding ${userIds.length} users' category interests...`);

  for (const userId of userIds) {
    const selected = pickRandom(CATEGORIES, 3 + Math.floor(Math.random() * 4));
    await db.insert(impressions).values(
      selected.map((categoryId) => ({
        userId,
        itemId: categoryId,
        itemType: "category" as const,
        action: "liked" as const,
      }))
    ).onConflictDoNothing();
  }

  console.log(`  → ${userIds.length} users' category interests seeded`);
}
