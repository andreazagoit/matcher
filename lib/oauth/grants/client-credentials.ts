/**
 * Client Credentials Grant
 * RFC 6749 §4.4
 * 
 * Used for M2M (machine-to-machine) authentication
 */

import { validateClientCredentials, clientSupportsGrant } from "@/lib/models/spaces/operations";
import { validateScopes } from "../config";
import { OAuthErrors } from "../errors";
import {
  generateAccessToken,
  hashToken,
  storeAccessToken,
} from "../tokens";

export interface ClientCredentialsRequest {
  clientId: string;
  clientSecret: string;
  scope?: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: "Bearer";
  expires_in: number;
  scope: string;
}

/**
 * Handle client_credentials grant
 */
export async function handleClientCredentials(
  request: ClientCredentialsRequest
): Promise<TokenResponse> {
  // 1. Validate client credentials
  const client = await validateClientCredentials(request.clientId, request.clientSecret);
  if (!client) {
    throw OAuthErrors.invalidClient("Invalid client credentials");
  }

  // 2. Check grant type is allowed
  if (!clientSupportsGrant(client, "client_credentials")) {
    throw OAuthErrors.unauthorizedClient("Client not authorized for client_credentials grant");
  }

  // 3. Validate scopes (dynamic - validated against all supported scopes)
  const requestedScope = request.scope || "read:profile";
  const { valid, scopes, invalid } = validateScopes(requestedScope);

  if (!valid) {
    throw OAuthErrors.invalidScope(`Invalid scopes: ${invalid.join(", ")}`);
  }

  const scope = scopes.join(" ");
  const ttl = parseInt(client.accessTokenTtl || "3600", 10);

  // 4. Generate access token (no refresh token for client_credentials)
  const { token: accessToken, jti, expiresAt } = generateAccessToken({
    clientId: client.clientId,
    scope,
    ttl,
  });

  // 5. Store token
  await storeAccessToken({
    tokenHash: hashToken(accessToken),
    jti,
    clientId: client.clientId,
    scope,
    expiresAt,
  });

  return {
    access_token: accessToken,
    token_type: "Bearer",
    expires_in: ttl,
    scope,
  };
}

