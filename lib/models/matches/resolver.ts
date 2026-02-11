import { getDailyMatches } from "@/lib/services/matching";
import { GraphQLError } from "graphql";

interface ResolverContext {
    user?: { id: string } | null;
}

export const matchResolvers = {
    Query: {
        dailyMatches: async (_: unknown, __: unknown, context: ResolverContext) => {
            if (!context.user) {
                throw new GraphQLError("Unauthorized", {
                    extensions: { code: "UNAUTHORIZED" },
                });
            }

            return await getDailyMatches(context.user.id);
        },
    },
};
