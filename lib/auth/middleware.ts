/**
 * Auth Middleware
 * Provides authentication context for API routes and GraphQL
 * 
 * Supports:
 * - OAuth: Bearer token from OAuth flow
 * - API Key: M2M access (sk_live_xxx)
 */

import { NextRequest } from "next/server";
import { validateApiKey } from "@/lib/models/apps/operations";
import { getUserById } from "@/lib/models/users/operations";
import { db } from "@/lib/db/drizzle";
import { accessTokens } from "@/lib/models/tokens/schema";
import { eq, and, gt, isNull } from "drizzle-orm";
import crypto from "crypto";
import type { User } from "@/lib/models/users/schema";

export interface AuthContext {
    isAuthenticated: boolean;
    authType: "oauth" | "api_key" | null;
    userId?: string;
    user?: User;
    scopes?: string[];
    appId?: string;
    fullAccess?: boolean; // For M2M API keys
}

/**
 * Hash a token for lookup
 */
function hashToken(token: string): string {
    return crypto.createHash("sha256").update(token).digest("hex");
}

/**
 * Get access token by its plain value
 */
async function getAccessTokenByValue(token: string) {
    const tokenHash = hashToken(token);
    return db.query.accessTokens.findFirst({
        where: and(
            eq(accessTokens.tokenHash, tokenHash),
            gt(accessTokens.expiresAt, new Date()),
            isNull(accessTokens.revokedAt)
        ),
    });
}

/**
 * Extract auth context from request
 */
export async function getAuthContext(req: NextRequest): Promise<AuthContext> {
    const authHeader = req.headers.get("authorization");

    if (authHeader?.startsWith("Bearer ")) {
        const token = authHeader.substring(7);

        // 1. Check if it's an M2M API key (sk_live_xxx)
        if (token.startsWith("sk_live_")) {
            const app = await validateApiKey(token);
            if (app) {
                return {
                    isAuthenticated: true,
                    authType: "api_key",
                    appId: app.id,
                    fullAccess: true,
                };
            }
        }

        // 2. Check if it's an OAuth access token
        const accessToken = await getAccessTokenByValue(token);
        if (accessToken && accessToken.userId) {
            const user = await getUserById(accessToken.userId);
            return {
                isAuthenticated: true,
                authType: "oauth",
                userId: accessToken.userId,
                user: user || undefined,
                scopes: accessToken.scope?.split(" ") || [],
                appId: accessToken.clientId,
                fullAccess: false,
            };
        }
    }

    // 3. Try API Key via X-API-Key header
    const apiKey = req.headers.get("x-api-key");
    if (apiKey) {
        const app = await validateApiKey(apiKey);
        if (app) {
            return {
                isAuthenticated: true,
                authType: "api_key",
                appId: app.id,
                fullAccess: true,
            };
        }
    }

    // Not authenticated
    return {
        isAuthenticated: false,
        authType: null,
    };
}

/**
 * Check if auth context has a specific scope
 */
export function hasScope(auth: AuthContext, scope: string): boolean {
    if (auth.fullAccess) return true;
    return auth.scopes?.includes(scope) ?? false;
}
