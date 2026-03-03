import {
  getUserItems,
  addUserItem,
  updateUserItem,
  deleteUserItem,
  reorderUserItems,
} from "./operations";
import type { GraphQLContext } from "@/lib/graphql/context";
import { GraphQLError } from "graphql";

function requireAuth(context: GraphQLContext) {
  if (!context.auth.user) {
    throw new GraphQLError("Authentication required", {
      extensions: { code: "UNAUTHENTICATED" },
    });
  }
  return context.auth.user;
}

export const userItemResolvers = {
  User: {
    userItems: async (parent: { id: string }) => {
      return getUserItems(parent.id);
    },
  },

  Mutation: {
    addUserItem: async (
      _: unknown,
      { input }: { input: { type: "photo" | "prompt"; promptKey?: string; content: string; displayOrder?: number } },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      return addUserItem(user.id, input);
    },

    updateUserItem: async (
      _: unknown,
      { itemId, input }: { itemId: string; input: { content?: string; promptKey?: string } },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      return updateUserItem(itemId, user.id, input);
    },

    deleteUserItem: async (
      _: unknown,
      { itemId }: { itemId: string },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      return deleteUserItem(itemId, user.id);
    },

    reorderUserItems: async (
      _: unknown,
      { itemIds }: { itemIds: string[] },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      return reorderUserItems(user.id, itemIds);
    },
  },
};
