import {
  getProfileItems,
  addProfileItem,
  updateProfileItem,
  deleteProfileItem,
  reorderProfileItems,
} from "./operations";
import type { GraphQLContext } from "@/lib/graphql/context";

class AuthError extends Error {
  constructor(message: string, public code: string = "UNAUTHENTICATED") {
    super(message);
    this.name = "AuthError";
  }
}

export const profileItemResolvers = {
  Query: {
    profileItems: async (_: unknown, { userId }: { userId: string }) => {
      return getProfileItems(userId);
    },
  },

  Mutation: {
    addProfileItem: async (
      _: unknown,
      { input }: { input: { type: "photo" | "prompt"; promptKey?: string; content: string; displayOrder?: number } },
      context: GraphQLContext
    ) => {
      if (!context.auth.user) throw new AuthError("Authentication required");
      return addProfileItem(context.auth.user.id, input);
    },

    updateProfileItem: async (
      _: unknown,
      { itemId, input }: { itemId: string; input: { content?: string; promptKey?: string } },
      context: GraphQLContext
    ) => {
      if (!context.auth.user) throw new AuthError("Authentication required");
      return updateProfileItem(itemId, context.auth.user.id, input);
    },

    deleteProfileItem: async (
      _: unknown,
      { itemId }: { itemId: string },
      context: GraphQLContext
    ) => {
      if (!context.auth.user) throw new AuthError("Authentication required");
      return deleteProfileItem(itemId, context.auth.user.id);
    },

    reorderProfileItems: async (
      _: unknown,
      { itemIds }: { itemIds: string[] },
      context: GraphQLContext
    ) => {
      if (!context.auth.user) throw new AuthError("Authentication required");
      return reorderProfileItems(context.auth.user.id, itemIds);
    },
  },
};
