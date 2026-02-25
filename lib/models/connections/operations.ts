import { db } from "@/lib/db/drizzle";
import { connections } from "./schema";
import { messages } from "../messages/schema";
import { eq, or, and, desc } from "drizzle-orm";

/**
 * Send a connection request to another user.
 * Creates a connection (status=pending) referencing a specific user item.
 */
export async function sendConnectionRequest(
  initiatorId: string,
  recipientId: string,
  targetUserItemId: string,
  initialMessage: string | null = null,
) {
  const existing = await db.query.connections.findFirst({
    where: or(
      and(
        eq(connections.initiatorId, initiatorId),
        eq(connections.recipientId, recipientId),
      ),
      and(
        eq(connections.initiatorId, recipientId),
        eq(connections.recipientId, initiatorId),
      ),
    ),
  });

  if (existing) {
    if (existing.status === "declined") {
      throw new Error("This connection was previously declined");
    }
    throw new Error("A connection already exists between these users");
  }

  const [connection] = await db
    .insert(connections)
    .values({
      initiatorId,
      recipientId,
      targetUserItemId,
      initialMessage,
      status: "pending",
    })
    .returning();

  if (initialMessage) {
    await db.insert(messages).values({
      connectionId: connection.id,
      senderId: initiatorId,
      content: initialMessage,
    });
  }

  await db
    .update(connections)
    .set({ lastMessageAt: new Date() })
    .where(eq(connections.id, connection.id));

  return connection;
}

/**
 * Accept or decline a connection request.
 * Returns the updated connection.
 */
export async function respondToRequest(
  connectionId: string,
  recipientId: string,
  accept: boolean,
) {
  const connection = await db.query.connections.findFirst({
    where: eq(connections.id, connectionId),
  });

  if (!connection) throw new Error("Connection not found");
  if (connection.recipientId !== recipientId) throw new Error("Not the recipient");
  if (connection.status !== "pending") throw new Error("Not a pending request");

  const newStatus = accept ? "accepted" : "declined";

  const [updated] = await db
    .update(connections)
    .set({ status: newStatus, updatedAt: new Date() })
    .where(eq(connections.id, connectionId))
    .returning();

  return updated;
}

/**
 * Get pending message requests for a user (inbox).
 */
export async function getMessageRequests(userId: string) {
  return db.query.connections.findMany({
    where: and(
      eq(connections.recipientId, userId),
      eq(connections.status, "pending"),
    ),
    orderBy: [desc(connections.createdAt)],
  });
}

/**
 * Get active connections for a user.
 */
export async function getActiveConnections(userId: string) {
  return db.query.connections.findMany({
    where: and(
      or(
        eq(connections.initiatorId, userId),
        eq(connections.recipientId, userId),
      ),
      eq(connections.status, "accepted"),
    ),
    orderBy: [desc(connections.lastMessageAt)],
  });
}

/**
 * Get a connection by ID, verifying the user is a participant.
 */
export async function getConnectionById(connectionId: string, userId: string) {
  const connection = await db.query.connections.findFirst({
    where: eq(connections.id, connectionId),
  });

  if (!connection) return null;
  if (connection.initiatorId !== userId && connection.recipientId !== userId) {
    return null;
  }

  return connection;
}

/**
 * Send a message in an existing connection.
 * Allowed if: connection is active, or sender is initiator and status is request.
 */
export async function sendMessage(
  connectionId: string,
  senderId: string,
  content: string,
) {
  const connection = await db.query.connections.findFirst({
    where: eq(connections.id, connectionId),
  });

  if (!connection) throw new Error("Connection not found");

  const isParticipant =
    connection.initiatorId === senderId || connection.recipientId === senderId;
  if (!isParticipant) throw new Error("Not a participant");

  const canSend =
    connection.status === "accepted" ||
    (connection.status === "pending" && connection.initiatorId === senderId);
  if (!canSend) throw new Error("Cannot send messages in this connection");

  const [message] = await db
    .insert(messages)
    .values({ connectionId, senderId, content })
    .returning();

  await db
    .update(connections)
    .set({ lastMessageAt: new Date(), updatedAt: new Date() })
    .where(eq(connections.id, connectionId));

  return message;
}

/**
 * Get messages for a connection.
 */
export async function getMessages(connectionId: string) {
  return db.query.messages.findMany({
    where: eq(messages.connectionId, connectionId),
    orderBy: [desc(messages.createdAt)],
  });
}
