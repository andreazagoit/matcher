import { type AuthContext } from "@/lib/auth/utils";

interface PlatformContext {
    auth: AuthContext;
}

export const platformResolvers = {
    Query: {
        health: () => "Platform API v1 is operational",
        me: (_parent: unknown, _args: unknown, context: PlatformContext) => {
            if (!context.auth.user) {
                return null;
            }
            return {
                id: context.auth.user.id,
                name: `${context.auth.user.givenName} ${context.auth.user.familyName}`,
                image: context.auth.user.image
            };
        }
    },
};
