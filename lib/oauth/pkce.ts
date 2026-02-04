/**
 * PKCE (Proof Key for Code Exchange)
 * RFC 7636
 */

import crypto from "crypto";

/**
 * Generate a code verifier (random string 43-128 chars)
 */
export function generateCodeVerifier(): string {
  return crypto.randomBytes(32).toString("base64url");
}

/**
 * Generate code challenge from verifier using S256
 */
export function generateCodeChallenge(verifier: string): string {
  const hash = crypto.createHash("sha256").update(verifier).digest();
  return hash.toString("base64url");
}

/**
 * Verify code challenge matches verifier
 * OAuth 2.1 requires S256 only (plain is deprecated)
 */
export function verifyCodeChallenge(
  verifier: string,
  challenge: string
): boolean {
  const expectedChallenge = generateCodeChallenge(verifier);
  return expectedChallenge === challenge;
}


