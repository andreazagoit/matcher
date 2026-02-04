/**
 * OAuth Callback for Internal Login
 * GET /api/auth/callback/internal
 * 
 * Handles the OAuth callback after user authorizes.
 * Exchanges code for token and sets session cookie.
 */

import { NextRequest } from "next/server";
import { cookies } from "next/headers";

export async function GET(request: NextRequest) {
    const searchParams = request.nextUrl.searchParams;
    const code = searchParams.get("code");
    const error = searchParams.get("error");
    const errorDescription = searchParams.get("error_description");

    // Handle errors
    if (error) {
        console.error("OAuth error:", error, errorDescription);
        return Response.redirect(new URL(`/login?error=${encodeURIComponent(errorDescription || error)}`, request.url));
    }

    if (!code) {
        return Response.redirect(new URL("/login?error=No authorization code", request.url));
    }

    // Get PKCE code verifier from cookie
    const cookieStore = await cookies();
    const codeVerifier = cookieStore.get("pkce_code_verifier")?.value;

    if (!codeVerifier) {
        console.error("No PKCE code verifier found");
        return Response.redirect(new URL("/login?error=PKCE verification failed", request.url));
    }

    try {
        // Exchange code for token
        const tokenResponse = await fetch(`${request.nextUrl.origin}/oauth/token`, {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
            },
            body: new URLSearchParams({
                grant_type: "authorization_code",
                code,
                redirect_uri: `${request.nextUrl.origin}/api/auth/callback/internal`,
                client_id: process.env.OAUTH_CLIENT_ID || "",
                client_secret: process.env.OAUTH_CLIENT_SECRET || "",
                code_verifier: codeVerifier,
            }),
        });

        if (!tokenResponse.ok) {
            const errorData = await tokenResponse.json();
            console.error("Token exchange failed:", errorData);
            return Response.redirect(new URL(`/login?error=${encodeURIComponent(errorData.error_description || "Token exchange failed")}`, request.url));
        }

        const tokens = await tokenResponse.json();

        // Get user info
        const userinfoResponse = await fetch(`${request.nextUrl.origin}/oauth/userinfo`, {
            headers: {
                Authorization: `Bearer ${tokens.access_token}`,
            },
        });

        if (!userinfoResponse.ok) {
            return Response.redirect(new URL("/login?error=Failed to get user info", request.url));
        }

        const userinfo = await userinfoResponse.json();

        // Set user_id cookie for session
        cookieStore.set("user_id", userinfo.sub, {
            httpOnly: true,
            secure: process.env.NODE_ENV === "production",
            sameSite: "lax",
            maxAge: 60 * 60 * 24 * 7, // 7 days
        });

        // Clear PKCE cookie
        cookieStore.delete("pkce_code_verifier");

        // Redirect to dashboard
        return Response.redirect(new URL("/dashboard", request.url));
    } catch (error) {
        console.error("Callback error:", error);
        return Response.redirect(new URL("/login?error=Authentication failed", request.url));
    }
}
