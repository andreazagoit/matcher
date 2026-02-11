/**
 * PKCE Cookie API
 * POST /api/auth/pkce
 * 
 * Stores PKCE code verifier in httpOnly cookie
 */

import { NextRequest } from "next/server";
import { cookies } from "next/headers";

export async function POST(request: NextRequest) {
    try {
        const { codeVerifier } = await request.json();

        if (!codeVerifier) {
            return Response.json({ error: "Missing codeVerifier" }, { status: 400 });
        }

        const cookieStore = await cookies();
        cookieStore.set("pkce_code_verifier", codeVerifier, {
            httpOnly: true,
            secure: process.env.NODE_ENV === "production",
            sameSite: "lax",
            maxAge: 600, // 10 minutes
        });

        return Response.json({ success: true });
    } catch (error) {
        console.error("PKCE cookie error:", error);
        return Response.json({ error: "Failed to set PKCE cookie" }, { status: 500 });
    }
}
