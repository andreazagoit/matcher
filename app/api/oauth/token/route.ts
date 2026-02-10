/**
 * OAuth 2.0 Token Endpoint
 * POST /api/oauth/token
 * RFC 6749 §3.2
 */

import { NextRequest } from "next/server";
import { OAuthError, OAuthErrors } from "@/lib/oauth/errors";
import { handleClientCredentials } from "@/lib/oauth/grants/client-credentials";
import { exchangeCodeForTokens } from "@/lib/oauth/grants/authorization-code";
import { handleRefreshToken } from "@/lib/oauth/grants/refresh-token";

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
    // Parse form data (RFC 6749 requires application/x-www-form-urlencoded)
    const formData = await request.formData();
    const grantType = formData.get("grant_type") as string;

    if (!grantType) {
      throw OAuthErrors.invalidRequest("grant_type is required");
    }

    // Get client credentials from Basic Auth or body
    const basicAuth = parseBasicAuth(request.headers.get("authorization"));
    const clientId = basicAuth?.clientId || (formData.get("client_id") as string);
    const clientSecret = basicAuth?.clientSecret || (formData.get("client_secret") as string);

    let response;

    switch (grantType) {
      case "client_credentials": {
        if (!clientId || !clientSecret) {
          throw OAuthErrors.invalidRequest("client_id and client_secret required");
        }

        response = await handleClientCredentials({
          clientId,
          clientSecret,
          scope: formData.get("scope") as string | undefined,
        });
        break;
      }

      case "authorization_code": {
        const code = formData.get("code") as string;
        const redirectUri = formData.get("redirect_uri") as string;
        const codeVerifier = formData.get("code_verifier") as string | undefined;

        if (!clientId || !code || !redirectUri) {
          throw OAuthErrors.invalidRequest("client_id, code, and redirect_uri required");
        }

        response = await exchangeCodeForTokens({
          clientId,
          clientSecret,
          code,
          redirectUri,
          codeVerifier,
        });
        break;
      }

      case "refresh_token": {
        const refreshToken = formData.get("refresh_token") as string;

        if (!clientId || !refreshToken) {
          throw OAuthErrors.invalidRequest("client_id and refresh_token required");
        }

        response = await handleRefreshToken({
          clientId,
          clientSecret,
          refreshToken,
          scope: formData.get("scope") as string | undefined,
        });
        break;
      }

      default:
        throw OAuthErrors.unsupportedGrantType(`Unsupported grant_type: ${grantType}`);
    }

    return new Response(JSON.stringify(response), {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
        "Pragma": "no-cache",
      },
    });
  } catch (error) {
    if (error instanceof OAuthError) {
      return error.toResponse();
    }

    console.error("OAuth token error:", error);
    return OAuthErrors.serverError("Internal server error").toResponse();
  }
}


