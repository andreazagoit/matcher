/**
 * OAuth 2.0 Error Responses
 * RFC 6749 §5.2 + §4.1.2.1
 */

export type OAuthErrorCode =
  // Authorization errors (§4.1.2.1)
  | "invalid_request"
  | "unauthorized_client"
  | "access_denied"
  | "unsupported_response_type"
  | "invalid_scope"
  | "server_error"
  | "temporarily_unavailable"
  // Token errors (§5.2)
  | "invalid_client"
  | "invalid_grant"
  | "unsupported_grant_type";

export class OAuthError extends Error {
  constructor(
    public code: OAuthErrorCode,
    public description: string,
    public statusCode: number = 400,
    public uri?: string
  ) {
    super(description);
    this.name = "OAuthError";
  }

  toJSON() {
    return {
      error: this.code,
      error_description: this.description,
      ...(this.uri && { error_uri: this.uri }),
    };
  }

  toResponse() {
    return new Response(JSON.stringify(this.toJSON()), {
      status: this.statusCode,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-store",
        "Pragma": "no-cache",
      },
    });
  }
}

// Pre-defined errors
export const OAuthErrors = {
  // Authorization errors
  invalidRequest: (description = "Invalid request") =>
    new OAuthError("invalid_request", description, 400),

  unauthorizedClient: (description = "Client not authorized") =>
    new OAuthError("unauthorized_client", description, 401),

  accessDenied: (description = "Access denied") =>
    new OAuthError("access_denied", description, 403),

  unsupportedResponseType: (description = "Unsupported response type") =>
    new OAuthError("unsupported_response_type", description, 400),

  invalidScope: (description = "Invalid scope") =>
    new OAuthError("invalid_scope", description, 400),

  serverError: (description = "Server error") =>
    new OAuthError("server_error", description, 500),

  temporarilyUnavailable: (description = "Service temporarily unavailable") =>
    new OAuthError("temporarily_unavailable", description, 503),

  // Token errors
  invalidClient: (description = "Invalid client credentials") =>
    new OAuthError("invalid_client", description, 401),

  invalidGrant: (description = "Invalid grant") =>
    new OAuthError("invalid_grant", description, 400),

  unsupportedGrantType: (description = "Unsupported grant type") =>
    new OAuthError("unsupported_grant_type", description, 400),
} as const;


