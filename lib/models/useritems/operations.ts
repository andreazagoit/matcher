import { db } from "@/lib/db/drizzle";
import { userItems } from "./schema";
import type { UserItem } from "./schema";
import { eq, asc, and, gte } from "drizzle-orm";
import { GraphQLError } from "graphql";

/** Fetch all items for a user, ordered by displayOrder. */
export async function getUserItems(userId: string): Promise<UserItem[]> {
  return db.query.userItems.findMany({
    where: eq(userItems.userId, userId),
    orderBy: [asc(userItems.displayOrder)],
  });
}

/** Add a new item at the end of the user's profile. */
export async function addUserItem(
  userId: string,
  input: { type: "photo" | "prompt"; promptKey?: string; content: string; displayOrder?: number }
): Promise<UserItem> {
  let order = input.displayOrder;

  if (order === undefined) {
    const existing = await db.query.userItems.findMany({
      where: eq(userItems.userId, userId),
      orderBy: [asc(userItems.displayOrder)],
    });
    order = existing.length;
  }

  const [item] = await db
    .insert(userItems)
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
): Promise<UserItem> {
  const existing = await db.query.userItems.findFirst({
    where: and(eq(userItems.id, itemId), eq(userItems.userId, userId)),
  });

  if (!existing) {
    throw new GraphQLError("Item not found or not owned by user", { extensions: { code: "NOT_FOUND" } });
  }

  const [updated] = await db
    .update(userItems)
    .set({
      ...(input.content !== undefined && { content: input.content }),
      ...(input.promptKey !== undefined && { promptKey: input.promptKey }),
      updatedAt: new Date(),
    })
    .where(eq(userItems.id, itemId))
    .returning();

  return updated;
}

/** Delete an item and reindex the remaining items' displayOrder. */
export async function deleteUserItem(itemId: string, userId: string): Promise<boolean> {
  const existing = await db.query.userItems.findFirst({
    where: and(eq(userItems.id, itemId), eq(userItems.userId, userId)),
  });

  if (!existing) {
    throw new GraphQLError("Item not found or not owned by user", { extensions: { code: "NOT_FOUND" } });
  }

  await db.delete(userItems).where(eq(userItems.id, itemId));

  // Reindex items after the deleted one
  const remaining = await db.query.userItems.findMany({
    where: and(
      eq(userItems.userId, userId),
      gte(userItems.displayOrder, existing.displayOrder)
    ),
    orderBy: [asc(userItems.displayOrder)],
  });

  for (let i = 0; i < remaining.length; i++) {
    await db
      .update(userItems)
      .set({ displayOrder: existing.displayOrder + i })
      .where(eq(userItems.id, remaining[i].id));
  }

  return true;
}

/** Reorder all items for a user. itemIds must contain all existing item IDs. */
export async function reorderUserItems(
  userId: string,
  itemIds: string[]
): Promise<UserItem[]> {
  const existing = await db.query.userItems.findMany({
    where: eq(userItems.userId, userId),
  });

  const existingIds = new Set(existing.map((i) => i.id));
  for (const id of itemIds) {
    if (!existingIds.has(id)) {
      throw new GraphQLError(`Item ${id} not found or not owned by user`, { extensions: { code: "NOT_FOUND" } });
    }
  }

  for (let i = 0; i < itemIds.length; i++) {
    await db
      .update(userItems)
      .set({ displayOrder: i, updatedAt: new Date() })
      .where(and(eq(userItems.id, itemIds[i]), eq(userItems.userId, userId)));
  }

  return getUserItems(userId);
}
