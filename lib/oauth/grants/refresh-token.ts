/**
 * Refresh Token Grant
 * RFC 6749 §6
 */

import { validateClientCredentials, clientSupportsGrant, getAppByClientId } from "@/lib/models/apps/operations";
import { validateScopes } from "../config";
import { OAuthErrors } from "../errors";
import {
  generateAccessToken,
  generateRefreshToken,
  hashToken,
  storeAccessToken,
  storeRefreshToken,
  findRefreshTokenByHash,
  revokeRefreshToken,
} from "../tokens";

export interface RefreshTokenRequest {
  clientId: string;
  clientSecret?: string;
  refreshToken: string;
  scope?: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: "Bearer";
  expires_in: number;
  refresh_token?: string;
  scope: string;
}

/**
 * Handle refresh_token grant
 */
export async function handleRefreshToken(
  request: RefreshTokenRequest
): Promise<TokenResponse> {
  // 1. Validate client
  const client = await validateClientCredentials(request.clientId, request.clientSecret);

  // For public clients, just verify client exists
  if (!client) {
    const publicClient = await getAppByClientId(request.clientId);
    if (!publicClient || !publicClient.isActive) {
      throw OAuthErrors.invalidClient("Invalid client credentials");
    }
  }

  const clientForGrant = client || await getAppByClientId(request.clientId);

  // 2. Check grant type
  if (!clientForGrant || !clientSupportsGrant(clientForGrant, "refresh_token")) {
    throw OAuthErrors.unauthorizedClient("Client not authorized for refresh_token grant");
  }

  // 3. Find and validate refresh token
  const tokenHash = hashToken(request.refreshToken);
  const storedToken = await findRefreshTokenByHash(tokenHash);

  if (!storedToken) {
    throw OAuthErrors.invalidGrant("Invalid or expired refresh token");
  }

  // 4. Verify token belongs to this client
  if (storedToken.clientId !== request.clientId) {
    throw OAuthErrors.invalidGrant("Refresh token was not issued to this client");
  }

  // 5. Validate scope (must be subset of original scope)
  let scope = storedToken.scope;

  if (request.scope) {
    const originalScopes = storedToken.scope.split(" ");
    const { valid, scopes, invalid } = validateScopes(request.scope);

    if (!valid) {
      throw OAuthErrors.invalidScope(`Invalid scopes: ${invalid.join(", ")}.`);
    }

    // Check that requested scopes are a subset of original scopes
    const notInOriginal = scopes.filter(s => !originalScopes.includes(s));
    if (notInOriginal.length > 0) {
      throw OAuthErrors.invalidScope(`Scopes ${notInOriginal.join(", ")} not in original grant.`);
    }

    scope = scopes.join(" ");
  }

  // 6. Revoke old refresh token (rotation)
  await revokeRefreshToken(storedToken.jti);

  // 7. Generate new tokens
  const accessTtl = parseInt(clientForGrant.accessTokenTtl || "3600", 10);
  const refreshTtl = parseInt(clientForGrant.refreshTokenTtl || "2592000", 10);

  const { token: accessToken, jti: accessJti, expiresAt: accessExpiresAt } = generateAccessToken({
    clientId: storedToken.clientId,
    userId: storedToken.userId ?? undefined,
    scope,
    ttl: accessTtl,
  });

  // 8. Store new access token
  const storedAccessToken = await storeAccessToken({
    tokenHash: hashToken(accessToken),
    jti: accessJti,
    clientId: storedToken.clientId,
    userId: storedToken.userId ?? undefined,
    scope,
    expiresAt: accessExpiresAt,
  });

  // 9. Generate new refresh token
  const { token: newRefreshToken, jti: refreshJti, expiresAt: refreshExpiresAt } = generateRefreshToken({
    clientId: storedToken.clientId,
    userId: storedToken.userId ?? undefined,
    scope,
    ttl: refreshTtl,
  });

  await storeRefreshToken({
    tokenHash: hashToken(newRefreshToken),
    jti: refreshJti,
    accessTokenId: storedAccessToken.id,
    clientId: storedToken.clientId,
    userId: storedToken.userId ?? undefined,
    scope,
    expiresAt: refreshExpiresAt,
  });

  return {
    access_token: accessToken,
    token_type: "Bearer",
    expires_in: accessTtl,
    refresh_token: newRefreshToken,
    scope,
  };
}


