import { db } from "@/lib/db/drizzle";
import { conversations } from "./schema";
import { messages } from "../messages/schema";
import { users } from "../users/schema";
import { eq, or, and, desc, ne, sql } from "drizzle-orm";
import { GraphQLError } from "graphql";

import { type AuthContext } from "@/lib/auth/utils";

interface ResolverContext {
    auth: AuthContext;
}

export const conversationResolvers = {
    Query: {
        conversations: async (_parent: unknown, _args: unknown, context: ResolverContext) => {
            if (!context.auth.user) throw new GraphQLError("Unauthorized");
            const userId = context.auth.user.id;

            const userConversations = await db.query.conversations.findMany({
                where: or(
                    eq(conversations.participant1Id, userId),
                    eq(conversations.participant2Id, userId)
                ),
                orderBy: [desc(conversations.lastMessageAt)],
            });

            return userConversations;
        },

        conversation: async (_parent: unknown, { id }: { id: string }, context: ResolverContext) => {
            if (!context.auth.user) throw new GraphQLError("Unauthorized");
            const userId = context.auth.user.id;

            const conversation = await db.query.conversations.findFirst({
                where: eq(conversations.id, id),
            });

            if (!conversation) return null;

            if (conversation.participant1Id !== userId && conversation.participant2Id !== userId) {
                throw new GraphQLError("Forbidden");
            }

            return conversation;
        },

        messages: async (_parent: unknown, { conversationId }: { conversationId: string }, context: ResolverContext) => {
            if (!context.auth.user) throw new GraphQLError("Unauthorized");
            const userId = context.auth.user.id;

            const conversation = await db.query.conversations.findFirst({
                where: eq(conversations.id, conversationId),
            });

            if (!conversation) throw new GraphQLError("Conversation not found");

            if (conversation.participant1Id !== userId && conversation.participant2Id !== userId) {
                throw new GraphQLError("Forbidden");
            }

            const chatMessages = await db.query.messages.findMany({
                where: eq(messages.conversationId, conversationId),
                orderBy: [desc(messages.createdAt)],
            });

            return chatMessages;
        },
    },

    Mutation: {
        startConversation: async (_parent: unknown, { targetUserId }: { targetUserId: string }, context: ResolverContext) => {
            if (!context.auth.user) throw new GraphQLError("Unauthorized");
            const userId = context.auth.user.id;

            if (userId === targetUserId) throw new GraphQLError("Cannot chat with yourself");

            const existing = await db.query.conversations.findFirst({
                where: or(
                    and(eq(conversations.participant1Id, userId), eq(conversations.participant2Id, targetUserId)),
                    and(eq(conversations.participant1Id, targetUserId), eq(conversations.participant2Id, userId))
                )
            });

            if (existing) return existing;

            const [newConv] = await db.insert(conversations).values({
                participant1Id: userId,
                participant2Id: targetUserId,
            }).returning();

            return newConv;
        },

        sendMessage: async (_parent: unknown, { conversationId, content }: { conversationId: string, content: string }, context: ResolverContext) => {
            if (!context.auth.user) throw new GraphQLError("Unauthorized");
            const userId = context.auth.user.id;

            const conversation = await db.query.conversations.findFirst({
                where: eq(conversations.id, conversationId),
            });

            if (!conversation) throw new GraphQLError("Conversation not found");

            if (conversation.participant1Id !== userId && conversation.participant2Id !== userId) {
                throw new GraphQLError("Forbidden");
            }

            const [newMessage] = await db.insert(messages).values({
                conversationId,
                senderId: userId,
                content,
            }).returning();

            await db.update(conversations)
                .set({ lastMessageAt: new Date(), updatedAt: new Date() })
                .where(eq(conversations.id, conversationId));

            return newMessage;
        },

        markAsRead: async (_parent: unknown, { conversationId }: { conversationId: string }, context: ResolverContext) => {
            if (!context.auth.user) throw new GraphQLError("Unauthorized");
            const userId = context.auth.user.id;

            await db.update(messages)
                .set({ readAt: new Date() })
                .where(and(
                    eq(messages.conversationId, conversationId),
                    ne(messages.senderId, userId),
                    sql`read_at IS NULL`
                ));

            return true;
        }
    },

    Conversation: {
        participant1: async (parent: { participant1Id: string }) => {
            return db.query.users.findFirst({ where: eq(users.id, parent.participant1Id) });
        },
        participant2: async (parent: { participant2Id: string }) => {
            return db.query.users.findFirst({ where: eq(users.id, parent.participant2Id) });
        },
        otherParticipant: async (parent: { participant1Id: string, participant2Id: string }, _args: unknown, context: ResolverContext) => {
            const myId = context.auth.user?.id;
            const otherId = parent.participant1Id === myId ? parent.participant2Id : parent.participant1Id;
            return db.query.users.findFirst({ where: eq(users.id, otherId) });
        },
        lastMessage: async (parent: { id: string }) => {
            return db.query.messages.findFirst({
                where: eq(messages.conversationId, parent.id),
                orderBy: [desc(messages.createdAt)]
            });
        },
        unreadCount: async (parent: { id: string }, _args: unknown, context: ResolverContext) => {
            const myId = context.auth.user?.id;
            if (!myId) return 0;

            const count = await db.$count(messages, and(
                eq(messages.conversationId, parent.id),
                ne(messages.senderId, myId),
                sql`read_at IS NULL`
            ));
            return count;
        }
    },

    Message: {
        sender: async (parent: { senderId: string }) => {
            return db.query.users.findFirst({ where: eq(users.id, parent.senderId) });
        }
    }
};
