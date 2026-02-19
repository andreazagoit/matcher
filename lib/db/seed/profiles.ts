import { db } from "../drizzle";
import { profiles } from "../../models/profiles/schema";
import { userInterests } from "../../models/interests/schema";
import { ALL_TAGS } from "../../models/tags/data";

function pickRandom<T>(arr: T[], count: number): T[] {
  const shuffled = [...arr].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, count);
}

/**
 * Seed profiles and interests for a list of user IDs.
 * Each user gets an empty profile (for future behaviorEmbedding)
 * and 5–10 random interests with weight=1.0.
 */
export async function seedProfiles(userIds: string[]) {
  console.log(`\n📋 Seeding ${userIds.length} profiles + interests...`);

  for (const userId of userIds) {
    await db.insert(profiles).values({ userId });

    const tags = pickRandom(ALL_TAGS, 5 + Math.floor(Math.random() * 6));
    for (const tag of tags) {
      await db.insert(userInterests).values({
        userId,
        tag,
        weight: 1.0,
      });
    }
  }

  console.log(`  → ${userIds.length} profiles + interests created`);
}
