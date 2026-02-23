import { db } from "../drizzle";
import { users } from "../../models/users/schema";
import { ALL_TAGS } from "../../models/tags/data";
import { eq } from "drizzle-orm";

function pickRandom<T>(arr: T[], count: number): T[] {
  return [...arr].sort(() => Math.random() - 0.5).slice(0, count);
}

/**
 * Seed tags for a list of user IDs.
 * Each user gets 5–10 random tags stored directly on users.tags.
 */
export async function seedProfiles(userIds: string[]) {
  console.log(`\n📋 Seeding ${userIds.length} users' tags...`);

  for (const userId of userIds) {
    const tags = pickRandom(ALL_TAGS, 5 + Math.floor(Math.random() * 6));
    await db
      .update(users)
      .set({ tags, updatedAt: new Date() })
      .where(eq(users.id, userId));
  }

  console.log(`  → ${userIds.length} users' tags seeded`);
}
