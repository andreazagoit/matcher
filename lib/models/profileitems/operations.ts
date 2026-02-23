import { db } from "@/lib/db/drizzle";
import { profileItems } from "./schema";
import type { ProfileItem } from "./schema";
import { eq, asc, and, gte } from "drizzle-orm";
import { GraphQLError } from "graphql";

/** Fetch all items for a user, ordered by displayOrder. */
export async function getUserItems(userId: string): Promise<ProfileItem[]> {
  return db.query.profileItems.findMany({
    where: eq(profileItems.userId, userId),
    orderBy: [asc(profileItems.displayOrder)],
  });
}

/** Add a new item at the end of the user's profile. */
export async function addUserItem(
  userId: string,
  input: { type: "photo" | "prompt"; promptKey?: string; content: string; displayOrder?: number }
): Promise<ProfileItem> {
  let order = input.displayOrder;

  if (order === undefined) {
    const existing = await db.query.profileItems.findMany({
      where: eq(profileItems.userId, userId),
      orderBy: [asc(profileItems.displayOrder)],
    });
    order = existing.length;
  }

  const [item] = await db
    .insert(profileItems)
    .values({
      userId,
      type: input.type,
      promptKey: input.promptKey ?? null,
      content: input.content,
      displayOrder: order,
    })
    .returning();

  return item;
}

/** Update content or promptKey of a specific item. */
export async function updateUserItem(
  itemId: string,
  userId: string,
  input: { content?: string; promptKey?: string }
): Promise<ProfileItem> {
  const existing = await db.query.profileItems.findFirst({
    where: and(eq(profileItems.id, itemId), eq(profileItems.userId, userId)),
  });

  if (!existing) {
    throw new GraphQLError("Item not found or not owned by user", { extensions: { code: "NOT_FOUND" } });
  }

  const [updated] = await db
    .update(profileItems)
    .set({
      ...(input.content !== undefined && { content: input.content }),
      ...(input.promptKey !== undefined && { promptKey: input.promptKey }),
      updatedAt: new Date(),
    })
    .where(eq(profileItems.id, itemId))
    .returning();

  return updated;
}

/** Delete an item and reindex the remaining items' displayOrder. */
export async function deleteUserItem(itemId: string, userId: string): Promise<boolean> {
  const existing = await db.query.profileItems.findFirst({
    where: and(eq(profileItems.id, itemId), eq(profileItems.userId, userId)),
  });

  if (!existing) {
    throw new GraphQLError("Item not found or not owned by user", { extensions: { code: "NOT_FOUND" } });
  }

  await db.delete(profileItems).where(eq(profileItems.id, itemId));

  // Reindex items after the deleted one
  const remaining = await db.query.profileItems.findMany({
    where: and(
      eq(profileItems.userId, userId),
      gte(profileItems.displayOrder, existing.displayOrder)
    ),
    orderBy: [asc(profileItems.displayOrder)],
  });

  for (let i = 0; i < remaining.length; i++) {
    await db
      .update(profileItems)
      .set({ displayOrder: existing.displayOrder + i })
      .where(eq(profileItems.id, remaining[i].id));
  }

  return true;
}

/** Reorder all items for a user. itemIds must contain all existing item IDs. */
export async function reorderUserItems(
  userId: string,
  itemIds: string[]
): Promise<ProfileItem[]> {
  const existing = await db.query.profileItems.findMany({
    where: eq(profileItems.userId, userId),
  });

  const existingIds = new Set(existing.map((i) => i.id));
  for (const id of itemIds) {
    if (!existingIds.has(id)) {
      throw new GraphQLError(`Item ${id} not found or not owned by user`, { extensions: { code: "NOT_FOUND" } });
    }
  }

  for (let i = 0; i < itemIds.length; i++) {
    await db
      .update(profileItems)
      .set({ displayOrder: i, updatedAt: new Date() })
      .where(and(eq(profileItems.id, itemIds[i]), eq(profileItems.userId, userId)));
  }

  return getUserItems(userId);
}
