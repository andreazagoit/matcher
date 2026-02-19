import { db } from "@/lib/db/drizzle";
import { messages } from "../messages/schema";
import { users } from "../users/schema";
import { eq, and, ne, desc, sql } from "drizzle-orm";
import { GraphQLError } from "graphql";
import type { GraphQLContext } from "@/lib/graphql/context";
import {
  sendMessageRequest,
  respondToRequest,
  getMessageRequests,
  getActiveConversations,
  getConversationById,
  sendMessage,
  getMessages,
} from "./operations";
import { boostInterestsFromTags } from "../interests/operations";
import { getSharedInterestTags } from "../interests/operations";
import type { Conversation } from "./schema";

function requireAuth(context: GraphQLContext) {
  if (!context.auth.user) {
    throw new GraphQLError("Authentication required", {
      extensions: { code: "UNAUTHENTICATED" },
    });
  }
  return context.auth.user;
}

export const conversationResolvers = {
  Query: {
    messageRequests: async (
      _: unknown,
      __: unknown,
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      return getMessageRequests(user.id);
    },

    conversations: async (
      _: unknown,
      __: unknown,
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      return getActiveConversations(user.id);
    },

    conversation: async (
      _: unknown,
      { id }: { id: string },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      return getConversationById(id, user.id);
    },

    messages: async (
      _: unknown,
      { conversationId }: { conversationId: string },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      const conversation = await getConversationById(conversationId, user.id);
      if (!conversation) {
        throw new GraphQLError("Conversation not found or access denied");
      }
      return getMessages(conversationId);
    },
  },

  Mutation: {
    sendMessageRequest: async (
      _: unknown,
      {
        recipientId,
        content,
        source,
      }: { recipientId: string; content: string; source?: string },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      if (user.id === recipientId) {
        throw new GraphQLError("Cannot message yourself");
      }

      try {
        return await sendMessageRequest(
          user.id,
          recipientId,
          content,
          (source as "discovery" | "event" | "space") || "discovery",
        );
      } catch (err) {
        throw new GraphQLError(
          err instanceof Error ? err.message : "Failed to send request",
        );
      }
    },

    respondToRequest: async (
      _: unknown,
      { conversationId, accept }: { conversationId: string; accept: boolean },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);

      try {
        const updated = await respondToRequest(conversationId, user.id, accept);

        // Boost shared interests when a request is accepted
        if (accept) {
          const otherId = updated.initiatorId;
          getSharedInterestTags(user.id, otherId)
            .then((sharedTags) => {
              if (sharedTags.length > 0) {
                Promise.all([
                  boostInterestsFromTags(user.id, sharedTags, 0.05),
                  boostInterestsFromTags(otherId, sharedTags, 0.05),
                ]);
              }
            })
            .catch(() => {});
        }

        return updated;
      } catch (err) {
        throw new GraphQLError(
          err instanceof Error ? err.message : "Failed to respond",
        );
      }
    },

    sendMessage: async (
      _: unknown,
      { conversationId, content }: { conversationId: string; content: string },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      try {
        return await sendMessage(conversationId, user.id, content);
      } catch (err) {
        throw new GraphQLError(
          err instanceof Error ? err.message : "Failed to send message",
        );
      }
    },

    markAsRead: async (
      _: unknown,
      { conversationId }: { conversationId: string },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      const conversation = await getConversationById(conversationId, user.id);
      if (!conversation) {
        throw new GraphQLError("Conversation not found");
      }

      await db
        .update(messages)
        .set({ readAt: new Date() })
        .where(
          and(
            eq(messages.conversationId, conversationId),
            ne(messages.senderId, user.id),
            sql`read_at IS NULL`,
          ),
        );

      return true;
    },
  },

  Conversation: {
    initiator: async (parent: Conversation) => {
      return db.query.users.findFirst({
        where: eq(users.id, parent.initiatorId),
      });
    },

    recipient: async (parent: Conversation) => {
      return db.query.users.findFirst({
        where: eq(users.id, parent.recipientId),
      });
    },

    otherUser: async (
      parent: Conversation,
      _: unknown,
      context: GraphQLContext,
    ) => {
      const myId = context.auth.user?.id;
      const otherId =
        parent.initiatorId === myId ? parent.recipientId : parent.initiatorId;
      return db.query.users.findFirst({ where: eq(users.id, otherId) });
    },

    lastMessage: async (parent: Conversation) => {
      return db.query.messages.findFirst({
        where: eq(messages.conversationId, parent.id),
        orderBy: [desc(messages.createdAt)],
      });
    },

    unreadCount: async (
      parent: Conversation,
      _: unknown,
      context: GraphQLContext,
    ) => {
      const myId = context.auth.user?.id;
      if (!myId) return 0;

      return db.$count(
        messages,
        and(
          eq(messages.conversationId, parent.id),
          ne(messages.senderId, myId),
          sql`read_at IS NULL`,
        ),
      );
    },
  },

  Message: {
    sender: async (parent: { senderId: string }) => {
      return db.query.users.findFirst({ where: eq(users.id, parent.senderId) });
    },
  },
};
