import { db } from "../../db/drizzle";
import { impressions } from "./schema";
import { CATEGORIES } from "../categories/data";

function pickRandom<T>(arr: T[], count: number): T[] {
  return [...arr].sort(() => Math.random() - 0.5).slice(0, count);
}

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
