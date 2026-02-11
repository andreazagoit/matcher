/**
 * OAuth Grants - Unified Types and Exports
 */

// Unified type for token responses
export interface TokenResponse {
    access_token: string;
    token_type: "Bearer";
    expires_in: number;
    refresh_token?: string;
    scope: string;
}

// Export specific functions to avoid name conflicts
export {
    validateAuthorizeRequest,
    createAuthCode,
    exchangeCodeForTokens,
    type AuthorizeRequest,
    type AuthorizeValidationResult,
    type TokenRequest,
} from "./authorization-code";

export {
    handleClientCredentials,
    type ClientCredentialsRequest,
} from "./client-credentials";

export {
    handleRefreshToken,
    type RefreshTokenRequest,
} from "./refresh-token";

