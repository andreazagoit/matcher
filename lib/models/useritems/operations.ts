import { db } from "@/lib/db/drizzle";
import { userItems } from "./schema";
import type { UserItem } from "./schema";
import { users } from "@/lib/models/users/schema";
import { eq, asc, and } from "drizzle-orm";

export { MIN_PHOTOS, MIN_PROMPTS, MAX_PHOTOS, MAX_PROMPTS } from "./validator";

/** Syncs users.image with the first photo userItem for the user. */
export async function syncUserProfileImage(userId: string): Promise<void> {
  const firstPhoto = await db.query.userItems.findFirst({
    where: and(eq(userItems.userId, userId), eq(userItems.type, "photo")),
    orderBy: [asc(userItems.displayOrder)],
  });
  await db
    .update(users)
    .set({ image: firstPhoto?.content ?? null })
    .where(eq(users.id, userId));
}

/** Fetch all items for a user, ordered by displayOrder within each type group. */
export async function getUserItems(userId: string): Promise<UserItem[]> {
  return db.query.userItems.findMany({
    where: eq(userItems.userId, userId),
    orderBy: [asc(userItems.displayOrder)],
  });
}
