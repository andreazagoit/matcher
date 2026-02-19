import { db } from "@/lib/db/drizzle";
import { notifications } from "./schema";
import type { Notification, NewNotification } from "./schema";
import { eq, and, desc, count } from "drizzle-orm";

export async function getNotificationsForUser(
  userId: string,
  limit = 20,
  offset = 0,
): Promise<Notification[]> {
  return db.query.notifications.findMany({
    where: eq(notifications.userId, userId),
    orderBy: [desc(notifications.createdAt)],
    limit,
    offset,
  });
}

export async function getUnreadCount(userId: string): Promise<number> {
  const [row] = await db
    .select({ count: count() })
    .from(notifications)
    .where(and(eq(notifications.userId, userId), eq(notifications.read, false)));
  return row?.count ?? 0;
}

export async function markAsRead(
  notificationId: string,
  userId: string,
): Promise<Notification | null> {
  const [updated] = await db
    .update(notifications)
    .set({ read: true })
    .where(
      and(
        eq(notifications.id, notificationId),
        eq(notifications.userId, userId),
      ),
    )
    .returning();
  return updated ?? null;
}

export async function markAllAsRead(userId: string): Promise<void> {
  await db
    .update(notifications)
    .set({ read: true })
    .where(
      and(eq(notifications.userId, userId), eq(notifications.read, false)),
    );
}

export async function createNotification(
  input: Omit<NewNotification, "id" | "createdAt" | "read">,
): Promise<Notification> {
  const [notification] = await db
    .insert(notifications)
    .values(input)
    .returning();
  return notification;
}

export async function deleteNotification(
  notificationId: string,
  userId: string,
): Promise<boolean> {
  const [deleted] = await db
    .delete(notifications)
    .where(
      and(
        eq(notifications.id, notificationId),
        eq(notifications.userId, userId),
      ),
    )
    .returning();
  return !!deleted;
}
