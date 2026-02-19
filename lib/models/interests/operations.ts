import { db } from "@/lib/db/drizzle";
import { userInterests, type UserInterest } from "./schema";
import { eq, and, sql } from "drizzle-orm";

/**
 * Get all interests for a user, ordered by weight descending.
 */
export async function getUserInterests(userId: string): Promise<UserInterest[]> {
  return db.query.userInterests.findMany({
    where: eq(userInterests.userId, userId),
    orderBy: (ui, { desc }) => [desc(ui.weight)],
  });
}

/**
 * Get interest tags as a flat list (for use in recommendations).
 */
export async function getUserInterestTags(userId: string): Promise<string[]> {
  const interests = await getUserInterests(userId);
  return interests.map((i) => i.tag);
}

/**
 * Set declared interests for a user.
 * Upserts each tag with weight=1.0.
 * Removes interests whose tags are no longer in the list.
 */
export async function setDeclaredInterests(
  userId: string,
  tags: string[],
): Promise<UserInterest[]> {
  const existing = await db.query.userInterests.findMany({
    where: eq(userInterests.userId, userId),
  });

  const existingTags = new Set(existing.map((e) => e.tag));
  const newTags = new Set(tags);

  const toRemove = existing.filter((e) => !newTags.has(e.tag));
  for (const item of toRemove) {
    await db.delete(userInterests).where(eq(userInterests.id, item.id));
  }

  for (const tag of tags) {
    if (existingTags.has(tag)) {
      await db
        .update(userInterests)
        .set({ weight: 1.0, updatedAt: new Date() })
        .where(
          and(eq(userInterests.userId, userId), eq(userInterests.tag, tag)),
        );
    } else {
      await db
        .insert(userInterests)
        .values({ userId, tag, weight: 1.0 })
        .onConflictDoUpdate({
          target: [userInterests.userId, userInterests.tag],
          set: { weight: 1.0, updatedAt: new Date() },
        });
    }
  }

  return getUserInterests(userId);
}

/**
 * Boost interest weight for a single tag.
 * If the interest doesn't exist, creates it with weight=delta.
 * Weight is capped at 1.0.
 */
export async function boostInterest(
  userId: string,
  tag: string,
  delta: number,
): Promise<void> {
  await db
    .insert(userInterests)
    .values({ userId, tag, weight: Math.min(1.0, delta) })
    .onConflictDoUpdate({
      target: [userInterests.userId, userInterests.tag],
      set: {
        weight: sql`LEAST(1.0, ${userInterests.weight} + ${delta})`,
        updatedAt: new Date(),
      },
    });
}

/**
 * Boost interest weights for multiple tags at once.
 */
export async function boostInterestsFromTags(
  userId: string,
  tags: string[],
  delta: number,
): Promise<void> {
  for (const tag of tags) {
    await boostInterest(userId, tag, delta);
  }
}

/**
 * Get shared interest tags between two users.
 */
export async function getSharedInterestTags(
  userIdA: string,
  userIdB: string,
): Promise<string[]> {
  const [interestsA, interestsB] = await Promise.all([
    getUserInterestTags(userIdA),
    getUserInterestTags(userIdB),
  ]);

  const setB = new Set(interestsB);
  return interestsA.filter((tag) => setB.has(tag));
}
