/**
 * OAuth 2.0 Token Introspection Endpoint
 * POST /oauth/api/introspect
 * RFC 7662
 */

import { NextRequest } from "next/server";
import { OAuthError, OAuthErrors } from "@/lib/oauth/errors";
import { validateClientCredentials } from "@/lib/models/spaces/operations";
import { verifyAccessToken, findAccessTokenByJti } from "@/lib/oauth/tokens";

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

    if (!token) {
      throw OAuthErrors.invalidRequest("token is required");
    }

    // Authenticate client (required for introspection)
    const basicAuth = parseBasicAuth(request.headers.get("authorization"));
    const clientId = basicAuth?.clientId || (formData.get("client_id") as string);
    const clientSecret = basicAuth?.clientSecret || (formData.get("client_secret") as string);

    if (!clientId || !clientSecret) {
      throw OAuthErrors.invalidClient("Client authentication required");
    }

    const client = await validateClientCredentials(clientId, clientSecret);
    if (!client) {
      throw OAuthErrors.invalidClient("Invalid client credentials");
    }

    // Verify token
    const decoded = verifyAccessToken(token);

    if (!decoded) {
      // Token invalid or expired
      return Response.json({ active: false });
    }

    // Check if revoked
    const storedToken = await findAccessTokenByJti(decoded.jti);
    if (!storedToken) {
      return Response.json({ active: false });
    }

    // Return introspection response
    return Response.json({
      active: true,
      scope: decoded.scope,
      client_id: decoded.client_id,
      username: decoded.user_id, // Optional
      token_type: "Bearer",
      exp: decoded.exp,
      iat: decoded.iat,
      sub: decoded.sub,
      aud: decoded.aud,
      iss: decoded.iss,
      jti: decoded.jti,
    });
  } catch (error) {
    if (error instanceof OAuthError) {
      return error.toResponse();
    }

    console.error("OAuth introspect error:", error);
    return OAuthErrors.serverError("Internal server error").toResponse();
  }
}


