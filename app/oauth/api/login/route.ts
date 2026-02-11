/**
 * User Login API
 * POST /api/auth/login
 */

import { NextRequest } from "next/server";
import { cookies } from "next/headers";
import { db } from "@/lib/db/drizzle";
import { users } from "@/lib/models/users/schema";
import { eq } from "drizzle-orm";

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const { email, password } = body;

        if (!email || !password) {
            return Response.json(
                { error: "Email and password are required" },
                { status: 400 }
            );
        }

        // Find user by email
        const user = await db.query.users.findFirst({
            where: eq(users.email, email),
        });

        if (!user) {
            return Response.json(
                { error: "Invalid credentials" },
                { status: 401 }
            );
        }

        // TODO: In production, verify password hash
        // For now, accept any password for demo purposes

        // Set user ID cookie
        const cookieStore = await cookies();
        cookieStore.set("user_id", user.id, {
            httpOnly: true,
            secure: process.env.NODE_ENV === "production",
            sameSite: "lax",
            maxAge: 60 * 60 * 24 * 7, // 7 days
        });

        return Response.json({
            user: {
                id: user.id,
                email: user.email,
                firstName: user.firstName,
                lastName: user.lastName,
            },
        });
    } catch (error) {
        console.error("Login error:", error);
        return Response.json(
            { error: error instanceof Error ? error.message : "Login failed" },
            { status: 400 }
        );
    }
}
