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
import { hasScope, requireAuth } from "@/lib/auth/middleware";
import type { CreateUserInput, UpdateUserInput } from "./validator";
import type { User } from "./schema";
import type { GraphQLContext } from "@/app/api/graphql/route";

interface MatchOptions {
  limit?: number;
  gender?: ("man" | "woman" | "non_binary")[];
  minAge?: number;
  maxAge?: number;
}

/**
 * GraphQL Error for authentication/authorization
 */
class AuthError extends Error {
  constructor(message: string, public code: string = "UNAUTHENTICATED") {
    super(message);
    this.name = "AuthError";
  }
}

export const userResolvers = {
  Query: {
    /**
     * Get a user by ID
     * Public or requires read:profile scope
     */
    user: async (
      _: unknown,
      { id }: { id: string },
      context: GraphQLContext
    ) => {
      // Allow if authenticated with any valid token/key
      if (!context.auth.isAuthenticated) {
        throw new AuthError("Authentication required");
      }
      return await getUserById(id);
    },

    /**
     * Get all users
     * Requires authentication (M2M or OAuth with scope)
     */
    users: async (_: unknown, __: unknown, context: GraphQLContext) => {
      if (!context.auth.isAuthenticated) {
        throw new AuthError("Authentication required");
      }
      
      // M2M has full access
      if (!context.auth.fullAccess && !hasScope(context.auth, "read:profile")) {
        throw new AuthError("Insufficient scope. Required: read:profile", "FORBIDDEN");
      }
      
      return await getAllUsers();
    },

    /**
     * Get current authenticated user
     * Requires OAuth token (not M2M - M2M doesn't have a "user")
     */
    me: async (_: unknown, __: unknown, context: GraphQLContext) => {
      if (!context.auth.isAuthenticated) {
        return null;
      }
      
      // M2M doesn't have a user context
      if (context.auth.authType === "api_key") {
        return null;
      }
      
      if (!context.auth.userId) {
        return null;
      }
      
      return context.auth.user || await getUserById(context.auth.userId);
    },

    /**
     * Find matching users
     * Requires read:matches scope or M2M access
     */
    findMatches: async (
      _: unknown,
      { userId, options }: { userId?: string; options?: MatchOptions },
      context: GraphQLContext
    ) => {
      if (!context.auth.isAuthenticated) {
        throw new AuthError("Authentication required");
      }
      
      // Check scope (M2M bypasses this via fullAccess)
      if (!hasScope(context.auth, "read:matches")) {
        throw new AuthError("Insufficient scope. Required: read:matches", "FORBIDDEN");
      }
      
      // Determine target user
      let targetUserId = userId;
      
      // If no userId provided and it's an OAuth user, use their ID
      if (!targetUserId && context.auth.authType === "oauth") {
        targetUserId = context.auth.userId;
      }
      
      if (!targetUserId) {
        throw new Error("User ID required. Provide userId parameter or authenticate as a user.");
      }
      
      const matches = await findMatches(targetUserId, {
        limit: options?.limit ?? 10,
        gender: options?.gender,
        minAge: options?.minAge,
        maxAge: options?.maxAge,
      });
      
      // Return only users
      return matches.map((match) => match.user);
    },
  },

  Mutation: {
    /**
     * Create a new user
     * Requires write:profile scope or M2M access
     */
    createUser: async (
      _: unknown,
      { input }: { input: CreateUserInput },
      context: GraphQLContext
    ) => {
      if (!context.auth.isAuthenticated) {
        throw new AuthError("Authentication required");
      }
      
      if (!hasScope(context.auth, "write:profile")) {
        throw new AuthError("Insufficient scope. Required: write:profile", "FORBIDDEN");
      }
      
      return await createUser(input);
    },

    /**
     * Update a user
     * Requires write:profile scope + must be own profile (or M2M)
     */
    updateUser: async (
      _: unknown,
      { id, input }: { id: string; input: UpdateUserInput },
      context: GraphQLContext
    ) => {
      if (!context.auth.isAuthenticated) {
        throw new AuthError("Authentication required");
      }
      
      if (!hasScope(context.auth, "write:profile")) {
        throw new AuthError("Insufficient scope. Required: write:profile", "FORBIDDEN");
      }
      
      // OAuth users can only update their own profile
      if (context.auth.authType === "oauth" && context.auth.userId !== id) {
        throw new AuthError("You can only update your own profile", "FORBIDDEN");
      }
      
      return await updateUser(id, input);
    },

    /**
     * Delete a user
     * Requires M2M access only (dangerous operation)
     */
    deleteUser: async (
      _: unknown,
      { id }: { id: string },
      context: GraphQLContext
    ) => {
      if (!context.auth.isAuthenticated) {
        throw new AuthError("Authentication required");
      }
      
      // Only M2M can delete users
      if (!context.auth.fullAccess) {
        throw new AuthError("This operation requires M2M API key access", "FORBIDDEN");
      }
      
      return await deleteUser(id);
    },
  },

  // Field resolvers
  User: {
    profile: async (user: User) => {
      return await getProfileByUserId(user.id);
    },
  },
};
