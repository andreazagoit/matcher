/**
 * OAuth 2.0 / OpenID Connect UserInfo Endpoint
 * GET/POST /oauth/api/userinfo
 * RFC 7662 / OpenID Connect Core 1.0 §5.3
 */

import { NextRequest } from "next/server";
import { OAuthErrors } from "@/lib/oauth/errors";
import { verifyAccessToken, findAccessTokenByJti } from "@/lib/oauth/tokens";
import { db } from "@/lib/db/drizzle";
import { eq } from "drizzle-orm";
import { users } from "@/lib/models/users/schema";

async function handleUserInfo(request: NextRequest) {
    try {
        // Extract Bearer token from Authorization header
        const authHeader = request.headers.get("authorization");
        if (!authHeader?.startsWith("Bearer ")) {
            return new Response(null, {
                status: 401,
                headers: {
                    "WWW-Authenticate": 'Bearer realm="userinfo", error="invalid_token"',
                },
            });
        }

        const token = authHeader.slice(7);

        // Verify access token
        const decoded = verifyAccessToken(token);
        if (!decoded) {
            return new Response(null, {
                status: 401,
                headers: {
                    "WWW-Authenticate": 'Bearer realm="userinfo", error="invalid_token", error_description="Token is invalid or expired"',
                },
            });
        }

        // Check if token is revoked
        const storedToken = await findAccessTokenByJti(decoded.jti);
        if (!storedToken) {
            return new Response(null, {
                status: 401,
                headers: {
                    "WWW-Authenticate": 'Bearer realm="userinfo", error="invalid_token", error_description="Token has been revoked"',
                },
            });
        }

        // Client credentials grant doesn't have a user
        if (!decoded.user_id) {
            return OAuthErrors.invalidRequest("Access token was not issued for a user").toResponse();
        }

        // Get user data
        const user = await db.query.users.findFirst({
            where: eq(users.id, decoded.user_id),
        });

        if (!user) {
            return OAuthErrors.invalidRequest("User not found").toResponse();
        }

        // Build response based on scopes
        const scopes = decoded.scope.split(" ");
        const response: Record<string, unknown> = {
            sub: user.id, // Always include subject
        };

        // OpenID scope - basic identity
        if (scopes.includes("openid")) {
            response.sub = user.id;
        }

        // Profile scope - basic profile info
        if (scopes.includes("profile") || scopes.includes("read:profile")) {
            response.name = `${user.firstName} ${user.lastName}`;
            response.given_name = user.firstName;
            response.family_name = user.lastName;
            if (user.gender) {
                response.gender = user.gender;
            }
            if (user.birthDate) {
                response.birthdate = user.birthDate;
            }
            response.created_at = user.createdAt;
            response.updated_at = user.updatedAt;
        }

        // Email scope
        if (scopes.includes("email")) {
            response.email = user.email;
            response.email_verified = true; // Assume verified if in DB
        }

        return Response.json(response, {
            headers: {
                "Cache-Control": "no-store",
            },
        });
    } catch (error) {
        console.error("UserInfo error:", error);
        return OAuthErrors.serverError("Internal server error").toResponse();
    }
}

export async function GET(request: NextRequest) {
    return handleUserInfo(request);
}

export async function POST(request: NextRequest) {
    return handleUserInfo(request);
}
