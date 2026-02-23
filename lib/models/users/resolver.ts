/**
 * User resolvers — CRUD and location.
 */

import {
  createUser,
  updateUser,
  deleteUser,
  getUserByUsername,
  getAllUsers,
  updateUserLocation,
  isUsernameTaken,
} from "./operations";
import { users } from "./schema";
import type { CreateUserInput, UpdateUserInput } from "./validator";
import type { GraphQLContext } from "@/lib/graphql/context";
import { embedUser, recommendTagsForUser } from "@/lib/models/embeddings/operations";
import { db } from "@/lib/db/drizzle";
import { eq } from "drizzle-orm";
import { isValidTag } from "@/lib/models/tags/data";
import { getUserItems } from "@/lib/models/profileitems/operations";
import type { ProfileItem } from "@/lib/models/profileitems/schema";
import { GraphQLError } from "graphql";

class AuthError extends Error {
  constructor(message: string, public code: string = "UNAUTHENTICATED") {
    super(message);
    this.name = "AuthError";
  }
}

export const userResolvers = {
  Query: {
    myTags: async (_: unknown, __: unknown, context: GraphQLContext) => {
      if (!context.auth.user) return [];
      const u = await db.query.users.findFirst({
        where: eq(users.id, context.auth.user.id),
        columns: { tags: true },
      });
      return u?.tags ?? [];
    },

    recommendedTags: async (
      _: unknown,
      { limit = 10 }: { limit?: number },
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) return [];
      const u = await db.query.users.findFirst({
        where: eq(users.id, context.auth.user.id),
        columns: { tags: true },
      });
      return recommendTagsForUser(context.auth.user.id, u?.tags ?? [], limit);
    },

    user: async (
      _: unknown,
      { username }: { username: string },
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
        await embedUser(id, {
          tags: updatedUser.tags ?? [],
          birthdate: updatedUser.birthdate ?? null,
          gender: updatedUser.gender ?? null,
          relationshipIntent: updatedUser.relationshipIntent ?? null,
          smoking: updatedUser.smoking ?? null,
          drinking: updatedUser.drinking ?? null,
          activityLevel: updatedUser.activityLevel ?? null,
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

    updateMyTags: async (
      _: unknown,
      { tags }: { tags: string[] },
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) {
        throw new AuthError("Authentication required");
      }

      const invalidTags = tags.filter((t) => !isValidTag(t));
      if (invalidTags.length > 0) {
        throw new GraphQLError(`Invalid tags: ${invalidTags.join(", ")}`);
      }

      const [updated] = await db
        .update(users)
        .set({ tags, updatedAt: new Date() })
        .where(eq(users.id, context.auth.user.id))
        .returning();

      if (!updated) throw new GraphQLError("User not found");

      // Regenerate embedding in background
      embedUser(context.auth.user.id, {
        tags,
        birthdate: updated.birthdate ?? null,
        gender: updated.gender ?? null,
        relationshipIntent: updated.relationshipIntent ?? null,
        smoking: updated.smoking ?? null,
        drinking: updated.drinking ?? null,
        activityLevel: updated.activityLevel ?? null,
      }).catch(() => {});

      return updated;
    },
  },

  User: {
    location: (parent: { location?: { x: number; y: number } | null }) => {
      if (!parent.location) return null;
      return { lat: parent.location.y, lon: parent.location.x };
    },
    tags: (parent: { tags?: string[] | null }) => parent.tags ?? [],
    userItems: (parent: { id: string }): Promise<ProfileItem[]> => {
      return getUserItems(parent.id);
    },
  },
};
