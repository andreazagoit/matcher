import { db } from "@/lib/db/drizzle";
import { messages } from "../messages/schema";
import { users } from "../users/schema";
import { eq, and, ne, desc, sql } from "drizzle-orm";
import { GraphQLError } from "graphql";
import type { GraphQLContext } from "@/lib/graphql/context";
import {
  sendConnectionRequest,
  respondToRequest,
  getMessageRequests,
  getActiveConnections,
  getConnectionById,
  sendMessage,
  getMessages,
} from "./operations";
import type { Connection } from "./schema";

function requireAuth(context: GraphQLContext) {
  if (!context.auth.user) {
    throw new GraphQLError("Authentication required", {
      extensions: { code: "UNAUTHENTICATED" },
    });
  }
  return context.auth.user;
}

export const connectionResolvers = {
  Query: {
    connectionRequests: async (
      _: unknown,
      __: unknown,
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      return getMessageRequests(user.id);
    },

    connections: async (
      _: unknown,
      __: unknown,
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      return getActiveConnections(user.id);
    },

    connection: async (
      _: unknown,
      { id }: { id: string },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      return getConnectionById(id, user.id);
    },

    messages: async (
      _: unknown,
      { connectionId }: { connectionId: string },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      const connection = await getConnectionById(connectionId, user.id);
      if (!connection) {
        throw new GraphQLError("Connection not found or access denied");
      }
      return getMessages(connectionId);
    },
  },

  Mutation: {
    sendConnectionRequest: async (
      _: unknown,
      {
        recipientId,
        targetUserItemId,
        initialMessage,
      }: { recipientId: string; targetUserItemId: string; initialMessage?: string },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      if (user.id === recipientId) {
        throw new GraphQLError("Cannot connect to yourself");
      }

      try {
        return await sendConnectionRequest(
          user.id,
          recipientId,
          targetUserItemId,
          initialMessage || null,
        );
      } catch (err) {
        throw new GraphQLError(
          err instanceof Error ? err.message : "Failed to send request",
        );
      }
    },

    respondToRequest: async (
      _: unknown,
      { connectionId, accept }: { connectionId: string; accept: boolean },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);

      try {
        const updated = await respondToRequest(connectionId, user.id, accept);


        return updated;
      } catch (err) {
        throw new GraphQLError(
          err instanceof Error ? err.message : "Failed to respond",
        );
      }
    },

    sendMessage: async (
      _: unknown,
      { connectionId, content }: { connectionId: string; content: string },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      try {
        return await sendMessage(connectionId, user.id, content);
      } catch (err) {
        throw new GraphQLError(
          err instanceof Error ? err.message : "Failed to send message",
        );
      }
    },

    markAsRead: async (
      _: unknown,
      { connectionId }: { connectionId: string },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      const connection = await getConnectionById(connectionId, user.id);
      if (!connection) {
        throw new GraphQLError("Connection not found");
      }

      await db
        .update(messages)
        .set({ readAt: new Date() })
        .where(
          and(
            eq(messages.connectionId, connectionId),
            ne(messages.senderId, user.id),
            sql`read_at IS NULL`,
          ),
        );

      return true;
    },
  },

  Connection: {
    initiator: async (parent: Connection) => {
      return db.query.users.findFirst({
        where: eq(users.id, parent.initiatorId),
      });
    },

    recipient: async (parent: Connection) => {
      return db.query.users.findFirst({
        where: eq(users.id, parent.recipientId),
      });
    },

    otherUser: async (
      parent: Connection,
      _: unknown,
      context: GraphQLContext,
    ) => {
      const myId = context.auth.user?.id;
      const otherId =
        parent.initiatorId === myId ? parent.recipientId : parent.initiatorId;
      return db.query.users.findFirst({ where: eq(users.id, otherId) });
    },

    lastMessage: async (parent: Connection) => {
      return db.query.messages.findFirst({
        where: eq(messages.connectionId, parent.id),
        orderBy: [desc(messages.createdAt)],
      });
    },

    unreadCount: async (
      parent: Connection,
      _: unknown,
      context: GraphQLContext,
    ) => {
      const myId = context.auth.user?.id;
      if (!myId) return 0;

      return db.$count(
        messages,
        and(
          eq(messages.connectionId, parent.id),
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
