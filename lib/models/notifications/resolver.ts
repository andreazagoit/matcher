import type { GraphQLContext } from "@/lib/graphql/context";
import { GraphQLError } from "graphql";
import {
  getNotificationsForUser,
  getUnreadCount,
  markAsRead,
  markAllAsRead,
  deleteNotification,
} from "./operations";

function requireAuth(context: GraphQLContext) {
  if (!context.auth.user) {
    throw new GraphQLError("Authentication required", {
      extensions: { code: "UNAUTHENTICATED" },
    });
  }
  return context.auth.user;
}

export const notificationResolvers = {
  Query: {
    notifications: async (
      _: unknown,
      { limit, offset }: { limit?: number; offset?: number },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      return getNotificationsForUser(user.id, limit ?? 20, offset ?? 0);
    },

    unreadNotificationsCount: async (
      _: unknown,
      __: unknown,
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      return getUnreadCount(user.id);
    },
  },

  Mutation: {
    markNotificationRead: async (
      _: unknown,
      { id }: { id: string },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      return markAsRead(id, user.id);
    },

    markAllNotificationsRead: async (
      _: unknown,
      __: unknown,
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      await markAllAsRead(user.id);
      return true;
    },

    deleteNotification: async (
      _: unknown,
      { id }: { id: string },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      return deleteNotification(id, user.id);
    },
  },
};
