/**
 * User Signup API
 * POST /api/auth/signup
 */

import { NextRequest } from "next/server";
import { cookies } from "next/headers";
import { signUp } from "@/lib/auth/service";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { email, password, firstName, lastName, birthDate, gender } = body;

    if (!email || !password || !firstName || !lastName || !birthDate) {
      return Response.json(
        { error: "All fields are required" },
        { status: 400 }
      );
    }

    // Validate password strength
    if (password.length < 8) {
      return Response.json(
        { error: "Password must be at least 8 characters" },
        { status: 400 }
      );
    }

    const { user, session } = await signUp({
      email,
      password,
      firstName,
      lastName,
      birthDate,
      gender,
    });

    // Set user ID cookie for OAuth flow
    const cookieStore = await cookies();
    cookieStore.set("user_id", user.id, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      maxAge: 60 * 60 * 24, // 24 hours
    });

    return Response.json({
      user: {
        id: user.id,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
      },
      session: session ? {
        access_token: session.access_token,
        expires_at: session.expires_at,
      } : null,
    });
  } catch (error) {
    console.error("Signup error:", error);
    return Response.json(
      { error: error instanceof Error ? error.message : "Signup failed" },
      { status: 400 }
    );
  }
}


