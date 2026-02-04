/**
 * OAuth 2.0 Server Configuration
 * RFC 6749, RFC 7636, RFC 8414
 */

export const OAUTH_CONFIG = {
  // Server info
  issuer: process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000",

  // Endpoints
  authorizationEndpoint: "/oauth/authorize",
  tokenEndpoint: "/oauth/token",
  revocationEndpoint: "/oauth/revoke",
  introspectionEndpoint: "/oauth/introspect",
  userinfoEndpoint: "/oauth/userinfo",

  // Supported features
  responseTypesSupported: ["code"],
  grantTypesSupported: [
    "authorization_code",
    "client_credentials",
    "refresh_token",
  ],
  tokenEndpointAuthMethodsSupported: [
    "client_secret_basic",
    "client_secret_post",
  ],
  codeChallengeMethodsSupported: ["S256"],

  // Token settings (defaults, can be overridden per client)
  accessTokenTtl: 3600, // 1 hour
  refreshTokenTtl: 2592000, // 30 days
  authorizationCodeTtl: 600, // 10 minutes

  // Scopes
  scopesSupported: [
    "openid",
    "profile",
    "email",
    "read:profile",
    "write:profile",
    "read:matches",
  ],
} as const;

export const SCOPES = {
  // Standard OpenID
  openid: "OpenID Connect",
  profile: "Read basic profile info",
  email: "Read email address",

  // App-specific
  "read:profile": "Read your profile",
  "write:profile": "Update your profile",
  "read:matches": "Find compatible users",
} as const;

export type Scope = keyof typeof SCOPES;

/**
 * Validate scope string against all supported scopes
 * Scopes are requested dynamically - no per-client restriction
 */
export function validateScopes(
  requestedScopes: string
): { valid: boolean; scopes: string[]; invalid: string[] } {
  const requested = requestedScopes.split(" ").filter(Boolean);
  const supportedScopes = Object.keys(SCOPES);
  const valid: string[] = [];
  const invalid: string[] = [];

  for (const scope of requested) {
    if (supportedScopes.includes(scope)) {
      valid.push(scope);
    } else {
      invalid.push(scope);
    }
  }

  return {
    valid: invalid.length === 0,
    scopes: valid,
    invalid,
  };
}

