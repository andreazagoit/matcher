/**
 * API Key Authentication Middleware
 * For M2M direct access using secret_key
 * 
 * Usage: Authorization: Bearer sk_live_xxx
 * 
 * M2M has FULL ACCESS - no scope restrictions
 */

import { NextRequest } from "next/server";
import { validateApiKey } from "@/lib/models/oauth-clients/operations";
import type { OAuthApp } from "@/lib/models/oauth-clients/schema";

export interface ApiKeyContext {
  type: "api_key";
  app: OAuthApp;
  /** M2M has full access, no scope restrictions */
  fullAccess: true;
}

/**
 * Validate API Key from request header
 * Returns app context if valid, null otherwise
 */
export async function validateApiKeyFromRequest(
  request: NextRequest
): Promise<ApiKeyContext | null> {
  const authHeader = request.headers.get("authorization");
  
  if (!authHeader?.startsWith("Bearer ")) {
    return null;
  }

  const token = authHeader.slice(7); // Remove "Bearer "
  
  // Check if it's an API key (starts with sk_)
  if (!token.startsWith("sk_")) {
    return null;
  }

  const app = await validateApiKey(token);
  
  if (!app) {
    return null;
  }

  // M2M has full access - no scope restrictions
  return {
    type: "api_key",
    app,
    fullAccess: true,
  };
}
