import { auth } from "@/lib/auth";
import { headers } from "next/headers";

export interface AuthContext {
    user: {
        id: string;
        name: string;
        email: string;
        birthdate: string;
        gender: string | null;
        createdAt: string;
        updatedAt: string;
        image: string | null;
    } | null;
}

/**
 * Utility to get the authenticated user context for GraphQL resolvers.
 * Uses better-auth session from cookies.
 */
export async function getAuthContext(): Promise<AuthContext> {
    const session = await auth.api.getSession({
        headers: await headers(),
    });

    if (!session?.user?.id) {
        return { user: null };
    }

    const u = session.user;

    return {
        user: {
            id: u.id,
            name: u.name || "",
            email: u.email || "",
            birthdate: (u as Record<string, unknown>).birthdate as string || "",
            gender: (u as Record<string, unknown>).gender as string | null || null,
            createdAt: u.createdAt ? new Date(u.createdAt).toISOString() : new Date().toISOString(),
            updatedAt: u.updatedAt ? new Date(u.updatedAt).toISOString() : new Date().toISOString(),
            image: u.image || null,
        },
    };
}
