/**
 * OAuth Grants - Unified Types and Exports
 */

// Tipo unificato per le risposte token
export interface TokenResponse {
    access_token: string;
    token_type: "Bearer";
    expires_in: number;
    refresh_token?: string;
    scope: string;
}

// Export funzioni specifiche per evitare conflitti
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

