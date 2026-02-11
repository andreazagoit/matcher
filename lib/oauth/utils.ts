/**
 * OAuth utility functions
 */

export interface AuthScope {
    scopes?: string[];
    fullAccess?: boolean;
}

/**
 * Check if auth context has a specific scope
 */
export function hasScope(auth: AuthScope, scope: string): boolean {
    if (auth.fullAccess) return true;
    return auth.scopes?.includes(scope) ?? false;
}
