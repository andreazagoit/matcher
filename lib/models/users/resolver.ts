/**
 * User resolvers — CRUD and location.
 */

import {
  createUser,
  updateUser,
  deleteUser,
  getUserById,
  getUserByUsername,
  getAllUsers,
  updateUserLocation,
  isUsernameTaken,
} from "./operations";
import type { CreateUserInput, UpdateUserInput } from "./validator";
import type { GraphQLContext } from "@/lib/graphql/context";
import { embedUser } from "@/lib/models/embeddings/operations";
import { getUserInterests } from "@/lib/models/interests/operations";
import type { UserInterest } from "@/lib/models/interests/schema";

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
      { username }: { username: string },
      _context: GraphQLContext,
    ) => {
      // Public profile lookup by username (used by /users/[username]).
      // Sensitive fields are still controlled at schema/query level.
      return await getUserByUsername(username);
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

    checkUsername: async (_: unknown, { username }: { username: string }) => {
      return await isUsernameTaken(username);
    },
  },

  Mutation: {
    createUser: async (
      _: unknown,
      { input }: { input: CreateUserInput },
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) {
        throw new AuthError("Authentication required");
      }
      return await createUser(input);
    },

    updateUser: async (
      _: unknown,
      { id, input }: { id: string; input: UpdateUserInput },
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) {
        throw new AuthError("Authentication required");
      }

      // Users can only update their own profile
      if (context.auth.user.id !== id) {
        throw new AuthError("You can only update your own profile", "FORBIDDEN");
      }

      const updatedUser = await updateUser(id, input);

      // Regenerate embedding in background when profile data changes
      (async () => {
        const interests = await getUserInterests(id);
        await embedUser(id, {
          tags: interests.map((i) => i.tag),
          birthdate: updatedUser.birthdate ?? null,
        });
      })().catch(() => {});

      return updatedUser;
    },

    deleteUser: async (
      _: unknown,
      { id }: { id: string },
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) {
        throw new AuthError("Authentication required");
      }

      if (context.auth.user.id !== id) {
        throw new AuthError("Forbidden", "FORBIDDEN");
      }

      return await deleteUser(id);
    },

    updateLocation: async (
      _: unknown,
      { lat, lon }: { lat: number; lon: number },
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) {
        throw new AuthError("Authentication required");
      }

      return await updateUserLocation(context.auth.user.id, lat, lon);
    },
  },

  User: {
    location: (parent: { location?: { x: number; y: number } | null }) => {
      if (!parent.location) return null;
      return { lat: parent.location.y, lon: parent.location.x };
    },
    interests: (parent: { id: string }): Promise<UserInterest[]> => {
      return getUserInterests(parent.id);
    },
  },
};
