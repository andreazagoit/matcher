/**
 * Resolvers for matching — local matching engine v2.
 */

import { GraphQLError } from "graphql";
import type { GraphQLContext } from "@/lib/graphql/context";
import { getUserInterests } from "@/lib/models/interests/operations";
import { getProfileByUserId } from "@/lib/models/profiles/operations";
import { findMatches } from "./operations";
import type { Gender } from "@/lib/graphql/__generated__/graphql";

function requireAuth(context: GraphQLContext) {
  if (!context.auth.user) {
    throw new GraphQLError("Authentication required", {
      extensions: { code: "UNAUTHENTICATED" },
    });
  }
  return context.auth.user;
}

export const matchResolvers = {
  Query: {
    findMatches: async (
      _: unknown,
      args: {
        maxDistance?: number;
        limit?: number;
        offset?: number;
        gender?: Gender[];
        minAge?: number;
        maxAge?: number;
      },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);

      return findMatches(user.id, {
        maxDistance: args.maxDistance ?? 50,
        limit: args.limit ?? 20,
        offset: args.offset ?? 0,
        gender: args.gender,
        minAge: args.minAge,
        maxAge: args.maxAge,
      });
    },

    profileStatus: async (
      _: unknown,
      __: unknown,
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      const [interests, profile] = await Promise.all([
        getUserInterests(user.id),
        getProfileByUserId(user.id),
      ]);

      return {
        hasProfile: interests.length > 0,
        updatedAt: profile?.updatedAt?.toISOString() ?? null,
      };
    },
  },
};
