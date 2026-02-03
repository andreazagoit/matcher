/**
 * Auth Middleware
 * Supports both:
 * - OAuth tokens (from authorization_code flow) - with scopes
 * - API Keys (for M2M direct access) - full access, no scopes
 */

import { NextRequest } from "next/server";
import { verifyAccessToken, findAccessTokenByJti } from "@/lib/oauth/tokens";
import { getUserById } from "@/lib/models/users/operations";
import { validateApiKeyFromRequest } from "./api-key";
import type { User } from "@/lib/models/users/schema";
import type { OAuthApp } from "@/lib/models/oauth-clients/schema";

export interface AuthContext {
  isAuthenticated: boolean;
  /** "oauth" for token-based, "api_key" for M2M */
  authType?: "oauth" | "api_key";
  /** M2M has full access, bypasses scope checks */
  fullAccess?: boolean;
  userId?: string;
  clientId?: string;
  /** Only for OAuth - M2M doesn't use scopes */
  scope?: string;
  scopes?: string[];
  user?: User;
  /** For API Key auth */
  app?: OAuthApp;
}

/**
 * Extract and validate auth from request
 * Supports both OAuth Bearer tokens and API Keys
 */
export async function getAuthContext(request: NextRequest): Promise<AuthContext> {
  const authHeader = request.headers.get("authorization");
  
  if (!authHeader?.startsWith("Bearer ")) {
    return { isAuthenticated: false };
  }

  const token = authHeader.slice(7);
  
  // Check if it's an API Key (M2M) - full access
  if (token.startsWith("sk_")) {
    const apiKeyContext = await validateApiKeyFromRequest(request);
    
    if (apiKeyContext) {
      return {
        isAuthenticated: true,
        authType: "api_key",
        fullAccess: true,
        clientId: apiKeyContext.app.clientId,
        app: apiKeyContext.app,
      };
    }
    
    return { isAuthenticated: false };
  }
  
  // Otherwise try OAuth token
  const decoded = verifyAccessToken(token);
  if (!decoded) {
    return { isAuthenticated: false };
  }

  // Check if revoked
  const storedToken = await findAccessTokenByJti(decoded.jti);
  if (!storedToken) {
    return { isAuthenticated: false };
  }

  // Get user if user_id present
  let user: User | undefined;
  if (decoded.user_id) {
    const foundUser = await getUserById(decoded.user_id);
    if (foundUser) {
      user = foundUser;
    }
  }

  const scopes = decoded.scope?.split(" ") || [];

  return {
    isAuthenticated: true,
    authType: "oauth",
    fullAccess: false,
    userId: decoded.user_id,
    clientId: decoded.client_id,
    scope: decoded.scope,
    scopes,
    user,
  };
}

/**
 * Check if scope is present in context
 * M2M (fullAccess) always returns true
 */
export function hasScope(context: AuthContext, requiredScope: string): boolean {
  // M2M has full access - bypass scope checks
  if (context.fullAccess) {
    return true;
  }
  
  if (context.scopes) {
    return context.scopes.includes(requiredScope);
  }
  if (!context.scope) return false;
  const scopes = context.scope.split(" ");
  return scopes.includes(requiredScope);
}

/**
 * Require authentication middleware
 */
export function requireAuth(context: AuthContext): void {
  if (!context.isAuthenticated) {
    throw new Error("Authentication required");
  }
}

/**
 * Require specific scope (M2M bypasses this)
 */
export function requireScope(context: AuthContext, scope: string): void {
  requireAuth(context);
  if (!hasScope(context, scope)) {
    throw new Error(`Missing required scope: ${scope}`);
  }
}
