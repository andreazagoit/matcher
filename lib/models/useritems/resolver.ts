import {
  getUserItems,
  addUserItem,
  updateUserItem,
  deleteUserItem,
  reorderUserItems,
} from "./operations";
import type { GraphQLContext } from "@/lib/graphql/context";

class AuthError extends Error {
  constructor(message: string, public code: string = "UNAUTHENTICATED") {
    super(message);
    this.name = "AuthError";
  }
}

export const userItemResolvers = {
  Query: {
    userItems: async (_: unknown, { userId }: { userId: string }) => {
      return getUserItems(userId);
    },
  },

  Mutation: {
    addUserItem: async (
      _: unknown,
      { input }: { input: { type: "photo" | "prompt"; promptKey?: string; content: string; displayOrder?: number } },
      context: GraphQLContext
    ) => {
      if (!context.auth.user) throw new AuthError("Authentication required");
      return addUserItem(context.auth.user.id, input);
    },

    updateUserItem: async (
      _: unknown,
      { itemId, input }: { itemId: string; input: { content?: string; promptKey?: string } },
      context: GraphQLContext
    ) => {
      if (!context.auth.user) throw new AuthError("Authentication required");
      return updateUserItem(itemId, context.auth.user.id, input);
    },

    deleteUserItem: async (
      _: unknown,
      { itemId }: { itemId: string },
      context: GraphQLContext
    ) => {
      if (!context.auth.user) throw new AuthError("Authentication required");
      return deleteUserItem(itemId, context.auth.user.id);
    },

    reorderUserItems: async (
      _: unknown,
      { itemIds }: { itemIds: string[] },
      context: GraphQLContext
    ) => {
      if (!context.auth.user) throw new AuthError("Authentication required");
      return reorderUserItems(context.auth.user.id, itemIds);
    },
  },
};
