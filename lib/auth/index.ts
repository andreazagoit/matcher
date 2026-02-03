export * from "./service";
export * from "./middleware";
export * from "./api-key";

import { NextRequest } from "next/server";
import { getAuthContext, type AuthContext } from "./middleware";

/**
 * Main auth middleware - handles both OAuth and API Key
 */
export async function authMiddleware(request: NextRequest): Promise<AuthContext> {
  return getAuthContext(request);
}

