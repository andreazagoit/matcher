import type { GraphQLContext } from "@/lib/graphql/context";
import { getDailyMatches } from "./operations";

export const matchResolvers = {
  User: {
    dailyMatches: async (
      parent: { id: string },
      _: unknown,
      context: GraphQLContext,
    ) => {
      if (context.auth.user?.id !== parent.id) return [];
      return getDailyMatches(parent.id);
    },
  },
};
