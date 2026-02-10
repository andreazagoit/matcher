import { getDailyMatches } from "@/lib/services/matching";
import { AuthContext } from "@/lib/auth/middleware";
import { GraphQLError } from "graphql";

export const matchResolvers = {
    Query: {
        dailyMatches: async (_: unknown, __: unknown, context: { auth: AuthContext }) => {
            if (!context.auth.isAuthenticated || !context.auth.user) {
                throw new GraphQLError("Unauthorized", {
                    extensions: { code: "UNAUTHORIZED" },
                });
            }

            return await getDailyMatches(context.auth.user.id);
        },
    },
};
