/**
 * OAuth 2.0 Server Configuration
 * RFC 6749, RFC 7636, RFC 8414
 */

export const OAUTH_CONFIG = {
  // Server info
  issuer: process.env.NEXT_PUBLIC_APP_URL!,

  // Endpoints
  authorizationEndpoint: "/oauth/authorize",
  tokenEndpoint: "/oauth/api/token",
  revocationEndpoint: "/oauth/api/revoke",
  introspectionEndpoint: "/oauth/api/introspect",
  userinfoEndpoint: "/oauth/api/userinfo",

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
  match: "Access matching functionality",
} as const;

/**
 * Scopes that require a completed profile/questionnaire
 */
export const SCOPES_REQUIRING_PROFILE = ["match", "read:matches"];

/**
 * Check if the requested scopes require a completed profile
 */
export function scopesRequireProfile(scopes: string): boolean {
  const requestedScopes = scopes.split(" ").filter(Boolean);
  return requestedScopes.some((s) => SCOPES_REQUIRING_PROFILE.includes(s));
}

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

