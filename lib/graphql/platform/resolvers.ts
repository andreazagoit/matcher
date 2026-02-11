interface PlatformUser {
    id: string;
    firstName: string;
    lastName: string;
    image?: string | null;
    [key: string]: any;
}

interface PlatformContext {
    user?: PlatformUser | null;
}

export const platformResolvers = {
    Query: {
        health: () => "Platform API v1 is operational",
        me: (_parent: unknown, _args: unknown, context: PlatformContext) => {
            if (!context.user) {
                return null;
            }
            return {
                id: context.user.id,
                name: `${context.user.firstName} ${context.user.lastName}`,
                image: context.user.image
            };
        }
    },
};
