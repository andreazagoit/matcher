/**
 * Auth Service
 * Handles user signup
 */

import { db } from "@/lib/db/drizzle";
import { users } from "@/lib/models/users/schema";

export interface SignUpInput {
    email: string;
    password: string;
    firstName: string;
    lastName: string;
    birthDate: string;
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
            firstName: input.firstName,
            lastName: input.lastName,
            birthDate: input.birthDate,
            gender: input.gender,
        })
        .returning();

    return {
        user,
        session: null, // No session - user should login via OAuth after signup
    };
}
