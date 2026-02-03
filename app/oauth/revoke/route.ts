/**
 * OAuth 2.0 Token Revocation Endpoint
 * POST /oauth/revoke
 * RFC 7009
 */

import { NextRequest } from "next/server";
import { OAuthError, OAuthErrors } from "@/lib/oauth/errors";
import { validateClientCredentials } from "@/lib/models/oauth-clients/operations";
import {
  verifyAccessToken,
  hashToken,
  revokeAccessToken,
  findRefreshTokenByHash,
  revokeRefreshToken,
} from "@/lib/oauth/tokens";

function parseBasicAuth(header: string | null): { clientId: string; clientSecret: string } | null {
  if (!header?.startsWith("Basic ")) return null;
  
  try {
    const decoded = Buffer.from(header.slice(6), "base64").toString();
    const [clientId, clientSecret] = decoded.split(":");
    if (clientId && clientSecret) {
      return { clientId, clientSecret };
    }
  } catch {
    // Invalid base64
  }
  return null;
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const token = formData.get("token") as string;
    const tokenTypeHint = formData.get("token_type_hint") as string | undefined;

    if (!token) {
      throw OAuthErrors.invalidRequest("token is required");
    }

    // Authenticate client
    const basicAuth = parseBasicAuth(request.headers.get("authorization"));
    const clientId = basicAuth?.clientId || (formData.get("client_id") as string);
    const clientSecret = basicAuth?.clientSecret || (formData.get("client_secret") as string);

    if (!clientId) {
      throw OAuthErrors.invalidClient("Client authentication required");
    }

    const client = await validateClientCredentials(clientId, clientSecret);
    if (!client) {
      throw OAuthErrors.invalidClient("Invalid client credentials");
    }

    // Try to revoke based on hint or try both
    if (tokenTypeHint === "refresh_token" || !tokenTypeHint) {
      const refreshTokenHash = hashToken(token);
      const refreshToken = await findRefreshTokenByHash(refreshTokenHash);
      
      if (refreshToken && refreshToken.clientId === clientId) {
        await revokeRefreshToken(refreshToken.jti);
        return new Response(null, { status: 200 });
      }
    }

    if (tokenTypeHint === "access_token" || !tokenTypeHint) {
      // Try as access token (JWT)
      const decoded = verifyAccessToken(token);
      if (decoded && decoded.client_id === clientId) {
        await revokeAccessToken(decoded.jti);
        return new Response(null, { status: 200 });
      }
    }

    // RFC 7009: Return 200 even if token not found (don't leak info)
    return new Response(null, { status: 200 });
  } catch (error) {
    if (error instanceof OAuthError) {
      return error.toResponse();
    }

    console.error("OAuth revoke error:", error);
    return OAuthErrors.serverError("Internal server error").toResponse();
  }
}

