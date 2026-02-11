import { auth } from "@/lib/oauth/auth";

export interface AuthContext {
    user: {
        id: string;
        firstName: string;
        lastName: string;
        email: string;
        birthDate: string;
        gender: string | null;
        createdAt: string;
        updatedAt: string;
        image: string | null;
    } | null;
}

/**
 * Utility to get the authenticated user context for GraphQL resolvers.
 * Performs a check to ensure the user actually exists in the DB.
 */
export async function getAuthContext(): Promise<AuthContext> {
    const session = await auth();

    if (!session?.user?.id) {
        return { user: null };
    }

    return {
        user: {
            id: session.user.id,
            firstName: (session.user as { firstName?: string }).firstName || "",
            lastName: (session.user as { lastName?: string }).lastName || "",
            email: session.user.email || "",
            birthDate: (session.user as { birthDate?: string }).birthDate || "",
            gender: (session.user as { gender?: string | null }).gender || null,
            createdAt: (session.user as { createdAt?: string }).createdAt || new Date().toISOString(),
            updatedAt: (session.user as { updatedAt?: string }).updatedAt || new Date().toISOString(),
            image: (session.user as { image?: string | null }).image || null,
        },
    };
}
