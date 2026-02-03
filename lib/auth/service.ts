/**
 * Auth Service
 * Handles user authentication via Supabase
 */

import { supabase } from "@/lib/db/supabase";
import { db } from "@/lib/db/drizzle";
import { users } from "@/lib/models/users/schema";
import { eq } from "drizzle-orm";

export interface SignUpInput {
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  birthDate: string;
  gender?: "man" | "woman" | "non_binary";
}

export interface LoginInput {
  email: string;
  password: string;
}

/**
 * Sign up a new user
 * Creates both Supabase auth user and local user record
 */
export async function signUp(input: SignUpInput) {
  // 1. Create Supabase auth user
  const { data: authData, error: authError } = await supabase.auth.signUp({
    email: input.email,
    password: input.password,
  });

  if (authError || !authData.user) {
    throw new Error(authError?.message || "Failed to create auth user");
  }

  // 2. Create local user record
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
    session: authData.session,
  };
}

/**
 * Login user
 */
export async function login(input: LoginInput) {
  const { data, error } = await supabase.auth.signInWithPassword({
    email: input.email,
    password: input.password,
  });

  if (error || !data.user) {
    throw new Error(error?.message || "Invalid credentials");
  }

  // Get local user
  const localUser = await db.query.users.findFirst({
    where: eq(users.email, input.email),
  });

  if (!localUser) {
    throw new Error("User not found");
  }

  return {
    user: localUser,
    session: data.session,
  };
}

/**
 * Logout user
 */
export async function logout() {
  const { error } = await supabase.auth.signOut();
  if (error) {
    throw new Error(error.message);
  }
}

/**
 * Get current Supabase session
 */
export async function getSession() {
  const { data, error } = await supabase.auth.getSession();
  if (error) {
    throw new Error(error.message);
  }
  return data.session;
}

/**
 * Verify Supabase token and get user
 */
export async function verifySupabaseToken(token: string) {
  const { data, error } = await supabase.auth.getUser(token);
  if (error || !data.user) {
    return null;
  }

  // Get local user
  const localUser = await db.query.users.findFirst({
    where: eq(users.email, data.user.email!),
  });

  return localUser;
}

