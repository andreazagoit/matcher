/**
 * Resolvers for matching — proxied to Identity Matcher's GraphQL API.
 */

import { GraphQLError } from "graphql";
import type { GraphQLContext } from "@/lib/graphql/context";
import { idmGraphQL, requireExternalUserId } from "@/lib/identitymatcher/client";
import {
  IDM_FIND_MATCHES,
  IDM_PROFILE_STATUS,
  IDM_ASSESSMENT_QUESTIONS,
  IDM_SUBMIT_ASSESSMENT,
} from "@/lib/identitymatcher/queries";

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
        limit?: number;
        gender?: string[];
        minAge?: number;
        maxAge?: number;
      },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      const externalId = await requireExternalUserId(user.id);

      const data = await idmGraphQL<{
        findMatches: unknown[];
      }>(IDM_FIND_MATCHES, {
        userId: externalId,
        limit: args.limit ?? 10,
        gender: args.gender,
        minAge: args.minAge,
        maxAge: args.maxAge,
      });

      return data.findMatches;
    },

    profileStatus: async (
      _: unknown,
      __: unknown,
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      const externalId = await requireExternalUserId(user.id);

      const data = await idmGraphQL<{
        profileStatus: unknown;
      }>(IDM_PROFILE_STATUS, { userId: externalId });

      return data.profileStatus;
    },

    assessmentQuestions: async () => {
      const data = await idmGraphQL<{
        assessmentQuestions: unknown[];
      }>(IDM_ASSESSMENT_QUESTIONS);

      return data.assessmentQuestions;
    },
  },

  Mutation: {
    submitAssessment: async (
      _: unknown,
      args: { answers: Record<string, unknown> },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      const externalId = await requireExternalUserId(user.id);

      const data = await idmGraphQL<{
        submitAssessment: unknown;
      }>(IDM_SUBMIT_ASSESSMENT, {
        userId: externalId,
        answers: args.answers,
      });

      return data.submitAssessment;
    },
  },
};
