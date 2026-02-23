/**
 * Resolvers for matching — local matching engine v2.
 */

import { GraphQLError } from "graphql";
import type { GraphQLContext } from "@/lib/graphql/context";
import { getStoredEmbedding } from "@/lib/models/embeddings/operations";
import { db } from "@/lib/db/drizzle";
import { users } from "@/lib/models/users/schema";
import { eq } from "drizzle-orm";
import { getUserItems } from "@/lib/models/profileitems/operations";
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
        limit: args.limit ?? 8,
        offset: args.offset ?? 0,
        // Keep "daily shuffle" only when caller does not request pagination.
        candidatePool: typeof args.offset === "number" ? undefined : 200,
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
      const [userRow, embeddingRow] = await Promise.all([
        db.query.users.findFirst({ where: eq(users.id, user.id), columns: { tags: true } }),
        getStoredEmbedding(user.id, "user"),
      ]);

      return {
        hasProfile: (userRow?.tags?.length ?? 0) > 0 || embeddingRow !== null,
        updatedAt: null,
      };
    },
  },
  MatchUser: {
    userItems: async (parent: { id: string }) => {
      return getUserItems(parent.id);
    },
  },
};
