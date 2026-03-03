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
  User: {
    notifications: async (
      parent: { id: string },
      { limit = 20, offset = 0 }: { limit?: number; offset?: number },
      context: GraphQLContext,
    ) => {
      if (context.auth.user?.id !== parent.id) {
        return { items: [], unreadCount: 0 };
      }
      const [items, unreadCount] = await Promise.all([
        getNotificationsForUser(parent.id, limit, offset),
        getUnreadCount(parent.id),
      ]);
      return { items, unreadCount };
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
