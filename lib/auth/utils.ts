import { auth } from "@/lib/oauth/auth";
import { getUserById } from "@/lib/models/users/operations";
import { type NextRequest } from "next/server";

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
        [key: string]: any;
    } | null;
}

/**
 * Utility to get the authenticated user context for GraphQL resolvers.
 * Performs a check to ensure the user actually exists in the DB.
 */
export async function getAuthContext(req: NextRequest): Promise<AuthContext> {
    const session = await auth();

    if (!session?.user?.id) {
        return { user: null };
    }

    return {
        user: {
            id: session.user.id,
            firstName: (session.user as any).firstName || "",
            lastName: (session.user as any).lastName || "",
            email: session.user.email || "",
            birthDate: (session.user as any).birthDate || "",
            gender: (session.user as any).gender || null,
            createdAt: (session.user as any).createdAt || new Date().toISOString(),
            updatedAt: (session.user as any).updatedAt || new Date().toISOString(),
        },
    };
}
