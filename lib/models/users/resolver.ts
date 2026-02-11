import {
  createUser,
  updateUser,
  deleteUser,
  getUserById,
  getAllUsers,
} from "./operations";
import {
  findMatches,
  getProfileByUserId,
} from "@/lib/models/profiles/operations";
import type { CreateUserInput, UpdateUserInput } from "./validator";
import type { User } from "./schema";
import type { GraphQLContext } from "@/app/api/client/v1/graphql/route";

interface MatchOptions {
  limit?: number;
  gender?: ("man" | "woman" | "non_binary")[];
  minAge?: number;
  maxAge?: number;
}

class AuthError extends Error {
  constructor(message: string, public code: string = "UNAUTHENTICATED") {
    super(message);
    this.name = "AuthError";
  }
}

export const userResolvers = {
  Query: {
    user: async (
      _: unknown,
      { id }: { id: string },
      context: GraphQLContext
    ) => {
      if (!context.auth.user) {
        throw new AuthError("Authentication required");
      }
      return await getUserById(id);
    },

    users: async (_: unknown, __: unknown, context: GraphQLContext) => {
      if (!context.auth.user) {
        throw new AuthError("Authentication required");
      }
      return await getAllUsers();
    },

    me: async (_: unknown, __: unknown, context: GraphQLContext) => {
      if (!context.auth.user) {
        return null;
      }
      return context.auth.user;
    },

    findMatches: async (
      _: unknown,
      { userId, options }: { userId?: string; options?: MatchOptions },
      context: GraphQLContext
    ) => {
      if (!context.auth.user) {
        throw new AuthError("Authentication required");
      }

      const targetUserId = userId || context.auth.user.id;

      if (!targetUserId) {
        throw new Error("User ID required.");
      }

      const matches = await findMatches(targetUserId, {
        limit: options?.limit ?? 10,
        gender: options?.gender,
        minAge: options?.minAge,
        maxAge: options?.maxAge,
      });

      return matches.map((match) => match.user);
    },
  },

  Mutation: {
    createUser: async (
      _: unknown,
      { input }: { input: CreateUserInput },
      context: GraphQLContext
    ) => {
      if (!context.auth.user) {
        throw new AuthError("Authentication required");
      }
      return await createUser(input);
    },

    updateUser: async (
      _: unknown,
      { id, input }: { id: string; input: UpdateUserInput },
      context: GraphQLContext
    ) => {
      if (!context.auth.user) {
        throw new AuthError("Authentication required");
      }

      // Users can only update their own profile
      if (context.auth.user.id !== id) {
        throw new AuthError("You can only update your own profile", "FORBIDDEN");
      }

      return await updateUser(id, input);
    },

    deleteUser: async (
      _: unknown,
      { id }: { id: string },
      context: GraphQLContext
    ) => {
      if (!context.auth.user) {
        throw new AuthError("Authentication required");
      }

      // Only allow self-deletion for now
      if (context.auth.user.id !== id) {
        throw new AuthError("Forbidden", "FORBIDDEN");
      }

      return await deleteUser(id);
    },
  },

  User: {
    profile: async (user: User) => {
      return await getProfileByUserId(user.id);
    },
  },
};
