import { AuthContext } from "@/lib/auth/middleware";

export const platformResolvers = {
    Query: {
        health: () => "Platform API v1 is operational",
        me: (parent: unknown, args: unknown, context: { auth: AuthContext }) => {
            if (!context.auth.isAuthenticated || !context.auth.user) {
                return null;
            }
            return {
                id: context.auth.user.id,
                name: `${context.auth.user.firstName} ${context.auth.user.lastName}`,
                image: context.auth.user.image
            };
        }
    },
};
