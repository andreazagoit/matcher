import { db } from "@/lib/db/drizzle";
import { conversations } from "./schema";
import { messages } from "../messages/schema";
import { eq, or, and, desc } from "drizzle-orm";

/**
 * Send a message request to another user.
 * Creates a conversation (status=request) and the first message.
 */
export async function sendMessageRequest(
  initiatorId: string,
  recipientId: string,
  content: string,
  source: "discovery" | "event" | "space" = "discovery",
) {
  const existing = await db.query.conversations.findFirst({
    where: or(
      and(
        eq(conversations.initiatorId, initiatorId),
        eq(conversations.recipientId, recipientId),
      ),
      and(
        eq(conversations.initiatorId, recipientId),
        eq(conversations.recipientId, initiatorId),
      ),
    ),
  });

  if (existing) {
    if (existing.status === "declined") {
      throw new Error("This conversation was previously declined");
    }
    throw new Error("A conversation already exists between these users");
  }

  const [conversation] = await db
    .insert(conversations)
    .values({
      initiatorId,
      recipientId,
      status: "request",
      source,
    })
    .returning();

  await db.insert(messages).values({
    conversationId: conversation.id,
    senderId: initiatorId,
    content,
  });

  await db
    .update(conversations)
    .set({ lastMessageAt: new Date() })
    .where(eq(conversations.id, conversation.id));

  return conversation;
}

/**
 * Accept or decline a message request.
 * Returns the updated conversation.
 */
export async function respondToRequest(
  conversationId: string,
  recipientId: string,
  accept: boolean,
) {
  const conversation = await db.query.conversations.findFirst({
    where: eq(conversations.id, conversationId),
  });

  if (!conversation) throw new Error("Conversation not found");
  if (conversation.recipientId !== recipientId) throw new Error("Not the recipient");
  if (conversation.status !== "request") throw new Error("Not a pending request");

  const newStatus = accept ? "active" : "declined";

  const [updated] = await db
    .update(conversations)
    .set({ status: newStatus, updatedAt: new Date() })
    .where(eq(conversations.id, conversationId))
    .returning();

  return updated;
}

/**
 * Get pending message requests for a user (inbox).
 */
export async function getMessageRequests(userId: string) {
  return db.query.conversations.findMany({
    where: and(
      eq(conversations.recipientId, userId),
      eq(conversations.status, "request"),
    ),
    orderBy: [desc(conversations.createdAt)],
  });
}

/**
 * Get active conversations for a user.
 */
export async function getActiveConversations(userId: string) {
  return db.query.conversations.findMany({
    where: and(
      or(
        eq(conversations.initiatorId, userId),
        eq(conversations.recipientId, userId),
      ),
      eq(conversations.status, "active"),
    ),
    orderBy: [desc(conversations.lastMessageAt)],
  });
}

/**
 * Get a conversation by ID, verifying the user is a participant.
 */
export async function getConversationById(conversationId: string, userId: string) {
  const conversation = await db.query.conversations.findFirst({
    where: eq(conversations.id, conversationId),
  });

  if (!conversation) return null;
  if (conversation.initiatorId !== userId && conversation.recipientId !== userId) {
    return null;
  }

  return conversation;
}

/**
 * Send a message in an existing conversation.
 * Allowed if: conversation is active, or sender is initiator and status is request.
 */
export async function sendMessage(
  conversationId: string,
  senderId: string,
  content: string,
) {
  const conversation = await db.query.conversations.findFirst({
    where: eq(conversations.id, conversationId),
  });

  if (!conversation) throw new Error("Conversation not found");

  const isParticipant =
    conversation.initiatorId === senderId || conversation.recipientId === senderId;
  if (!isParticipant) throw new Error("Not a participant");

  const canSend =
    conversation.status === "active" ||
    (conversation.status === "request" && conversation.initiatorId === senderId);
  if (!canSend) throw new Error("Cannot send messages in this conversation");

  const [message] = await db
    .insert(messages)
    .values({ conversationId, senderId, content })
    .returning();

  await db
    .update(conversations)
    .set({ lastMessageAt: new Date(), updatedAt: new Date() })
    .where(eq(conversations.id, conversationId));

  return message;
}

/**
 * Get messages for a conversation.
 */
export async function getMessages(conversationId: string) {
  return db.query.messages.findMany({
    where: eq(messages.conversationId, conversationId),
    orderBy: [desc(messages.createdAt)],
  });
}
