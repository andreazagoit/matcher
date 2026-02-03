/**
 * OAuth 2.0 Authorization Server Metadata
 * GET /.well-known/oauth-authorization-server
 * RFC 8414
 */

import { OAUTH_CONFIG } from "@/lib/oauth/config";

export async function GET() {
  const baseUrl = OAUTH_CONFIG.issuer;

  const metadata = {
    issuer: baseUrl,
    authorization_endpoint: `${baseUrl}${OAUTH_CONFIG.authorizationEndpoint}`,
    token_endpoint: `${baseUrl}${OAUTH_CONFIG.tokenEndpoint}`,
    revocation_endpoint: `${baseUrl}${OAUTH_CONFIG.revocationEndpoint}`,
    introspection_endpoint: `${baseUrl}${OAUTH_CONFIG.introspectionEndpoint}`,
    userinfo_endpoint: `${baseUrl}${OAUTH_CONFIG.userinfoEndpoint}`,
    
    response_types_supported: OAUTH_CONFIG.responseTypesSupported,
    grant_types_supported: OAUTH_CONFIG.grantTypesSupported,
    token_endpoint_auth_methods_supported: OAUTH_CONFIG.tokenEndpointAuthMethodsSupported,
    code_challenge_methods_supported: OAUTH_CONFIG.codeChallengeMethodsSupported,
    scopes_supported: OAUTH_CONFIG.scopesSupported,
    
    // Additional metadata
    service_documentation: `${baseUrl}/docs`,
    ui_locales_supported: ["en", "it"],
  };

  return Response.json(metadata, {
    headers: {
      "Cache-Control": "public, max-age=3600",
    },
  });
}

