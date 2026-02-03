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
 */
export function verifyCodeChallenge(
  verifier: string,
  challenge: string,
  method: "S256" | "plain" = "S256"
): boolean {
  if (method === "plain") {
    return verifier === challenge;
  }

  // S256
  const expectedChallenge = generateCodeChallenge(verifier);
  return expectedChallenge === challenge;
}


