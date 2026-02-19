import { db } from "@/lib/db/drizzle";
import { profiles, type Profile } from "./schema";
import { eq } from "drizzle-orm";

/**
 * Get a user's profile.
 */
export async function getProfileByUserId(
  userId: string,
): Promise<Profile | null> {
  const result = await db.query.profiles.findFirst({
    where: eq(profiles.userId, userId),
  });
  return result ?? null;
}

/**
 * Update the behavioral embedding (centroid of attended event embeddings).
 */
export async function updateBehaviorEmbedding(
  userId: string,
  behaviorEmbedding: number[],
): Promise<Profile> {
  const [updated] = await db
    .update(profiles)
    .set({
      behaviorEmbedding,
      updatedAt: new Date(),
    })
    .where(eq(profiles.userId, userId))
    .returning();

  if (!updated) throw new Error("Profile not found");
  return updated;
}
