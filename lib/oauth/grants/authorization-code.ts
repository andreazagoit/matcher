/**
 * Authorization Code Grant
 * RFC 6749 §4.1 + RFC 7636 (PKCE)
 */

import {
  validateClientCredentials,
  clientSupportsGrant,
  isRedirectUriAllowed,
  getClientByClientId,
} from "@/lib/models/oauth-clients/operations";
import {
  createAuthorizationCode,
  findAuthorizationCode,
  markCodeAsUsed,
} from "@/lib/models/oauth-codes/operations";
import { validateScopes } from "../config";
import { OAuthErrors } from "../errors";
import { verifyCodeChallenge } from "../pkce";
import {
  generateAccessToken,
  generateRefreshToken,
  hashToken,
  storeAccessToken,
  storeRefreshToken,
} from "../tokens";

// ============================================
// AUTHORIZE REQUEST
// ============================================

export interface AuthorizeRequest {
  responseType: string;
  clientId: string;
  redirectUri: string;
  scope?: string;
  state?: string;
  codeChallenge?: string;
  codeChallengeMethod?: "S256" | "plain";
}

export interface AuthorizeValidationResult {
  client: Awaited<ReturnType<typeof getClientByClientId>>;
  scope: string;
  redirectUri: string;
  state?: string;
  codeChallenge?: string;
  codeChallengeMethod?: "S256" | "plain";
}

/**
 * Validate authorization request
 */
export async function validateAuthorizeRequest(
  request: AuthorizeRequest
): Promise<AuthorizeValidationResult> {
  // 1. Validate response_type
  if (request.responseType !== "code") {
    throw OAuthErrors.unsupportedResponseType("Only 'code' response type is supported");
  }

  // 2. Validate client
  const client = await getClientByClientId(request.clientId);
  if (!client || !client.isActive) {
    throw OAuthErrors.invalidRequest("Invalid client_id");
  }

  // 3. Check grant type
  if (!clientSupportsGrant(client, "authorization_code")) {
    throw OAuthErrors.unauthorizedClient("Client not authorized for authorization_code grant");
  }

  // 4. Validate redirect_uri
  if (!isRedirectUriAllowed(client, request.redirectUri)) {
    throw OAuthErrors.invalidRequest("Invalid redirect_uri");
  }

  // 5. Validate scope (dynamic - validated against all supported scopes)
  const requestedScope = request.scope || "openid profile";
  const { valid, scopes, invalid } = validateScopes(requestedScope);
  
  if (!valid) {
    throw OAuthErrors.invalidScope(`Invalid scopes: ${invalid.join(", ")}`);
  }

  // 6. PKCE validation (always required for security)
  if (!request.codeChallenge) {
    throw OAuthErrors.invalidRequest("PKCE code_challenge is required");
  }

  if (request.codeChallengeMethod && request.codeChallengeMethod !== "S256" && request.codeChallengeMethod !== "plain") {
    throw OAuthErrors.invalidRequest("Invalid code_challenge_method. Use 'S256' or 'plain'");
  }

  return {
    client,
    scope: scopes.join(" "),
    redirectUri: request.redirectUri,
    state: request.state,
    codeChallenge: request.codeChallenge,
    codeChallengeMethod: request.codeChallengeMethod || "S256",
  };
}

/**
 * Create authorization code after user consent
 */
export async function createAuthCode(params: {
  clientId: string;
  userId: string;
  redirectUri: string;
  scope: string;
  state?: string;
  codeChallenge?: string;
  codeChallengeMethod?: "S256" | "plain";
}): Promise<string> {
  const authCode = await createAuthorizationCode(params);
  return authCode.code;
}

// ============================================
// TOKEN REQUEST
// ============================================

export interface TokenRequest {
  clientId: string;
  clientSecret?: string;
  code: string;
  redirectUri: string;
  codeVerifier?: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: "Bearer";
  expires_in: number;
  refresh_token?: string;
  scope: string;
}

/**
 * Exchange authorization code for tokens
 */
export async function exchangeCodeForTokens(
  request: TokenRequest
): Promise<TokenResponse> {
  // 1. Find authorization code
  const authCode = await findAuthorizationCode(request.code);
  if (!authCode) {
    throw OAuthErrors.invalidGrant("Invalid or expired authorization code");
  }

  // 2. Validate client exists
  const client = await getClientByClientId(request.clientId);
  if (!client || !client.isActive) {
    throw OAuthErrors.invalidClient("Invalid client");
  }

  // 3. Validate code belongs to this client
  if (authCode.clientId !== request.clientId) {
    throw OAuthErrors.invalidGrant("Authorization code was not issued to this client");
  }

  // 4. Validate redirect_uri matches
  if (authCode.redirectUri !== request.redirectUri) {
    throw OAuthErrors.invalidGrant("redirect_uri mismatch");
  }

  // 5. Validate PKCE
  if (authCode.codeChallenge) {
    if (!request.codeVerifier) {
      throw OAuthErrors.invalidGrant("code_verifier required");
    }

    const validPkce = verifyCodeChallenge(
      request.codeVerifier,
      authCode.codeChallenge,
      (authCode.codeChallengeMethod as "S256" | "plain") || "S256"
    );

    if (!validPkce) {
      throw OAuthErrors.invalidGrant("Invalid code_verifier");
    }
  }

  // 6. Mark code as used (before generating tokens to prevent replay)
  await markCodeAsUsed(request.code);

  // 7. Get client for TTL settings
  const clientForTtl = client || await getClientByClientId(request.clientId);
  const accessTtl = parseInt(clientForTtl?.accessTokenTtl || "3600", 10);
  const refreshTtl = parseInt(clientForTtl?.refreshTokenTtl || "2592000", 10);

  // 8. Generate tokens
  const { token: accessToken, jti: accessJti, expiresAt: accessExpiresAt } = generateAccessToken({
    clientId: authCode.clientId,
    userId: authCode.userId,
    scope: authCode.scope,
    ttl: accessTtl,
  });

  // 9. Store access token
  const storedAccessToken = await storeAccessToken({
    tokenHash: hashToken(accessToken),
    jti: accessJti,
    clientId: authCode.clientId,
    userId: authCode.userId,
    scope: authCode.scope,
    expiresAt: accessExpiresAt,
  });

  // 10. Generate and store refresh token (always for authorization_code flow)
  const { token: refreshToken, jti: refreshJti, expiresAt: refreshExpiresAt } = generateRefreshToken({
    clientId: authCode.clientId,
    userId: authCode.userId,
    scope: authCode.scope,
    ttl: refreshTtl,
  });

  await storeRefreshToken({
    tokenHash: hashToken(refreshToken),
    jti: refreshJti,
    accessTokenId: storedAccessToken.id,
    clientId: authCode.clientId,
    userId: authCode.userId,
    scope: authCode.scope,
    expiresAt: refreshExpiresAt,
  });

  return {
    access_token: accessToken,
    token_type: "Bearer",
    expires_in: accessTtl,
    refresh_token: refreshToken,
    scope: authCode.scope,
  };
}

