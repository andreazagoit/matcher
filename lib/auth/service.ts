/**
 * Auth Service
 * Handles user signup
 */

import { db } from "@/lib/db/drizzle";
import { users } from "@/lib/models/users/schema";

export interface SignUpInput {
    email: string;
    password: string;
    givenName: string;
    familyName: string;
    birthdate: string;
    gender?: "man" | "woman" | "non_binary";
}

/**
 * Sign up a new user
 * Creates a local user record
 * Note: Password handling should be done by the OAuth provider
 */
export async function signUp(input: SignUpInput) {
    // Create local user record
    const [user] = await db
        .insert(users)
        .values({
            email: input.email,
            givenName: input.givenName,
            familyName: input.familyName,
            birthdate: input.birthdate,
            gender: input.gender,
        })
        .returning();

    return {
        user,
        session: null, // No session - user should login via OAuth after signup
    };
}
