/**
 * User Login API
 * POST /api/auth/login
 */

import { NextRequest } from "next/server";
import { cookies } from "next/headers";
import { login } from "@/lib/auth/service";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { email, password } = body;

    if (!email || !password) {
      return Response.json(
        { error: "Email and password required" },
        { status: 400 }
      );
    }

    const { user, session } = await login({ email, password });

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
    console.error("Login error:", error);
    return Response.json(
      { error: error instanceof Error ? error.message : "Login failed" },
      { status: 401 }
    );
  }
}


