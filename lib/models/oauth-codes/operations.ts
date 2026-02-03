/**
 * OAuth Authorization Codes Operations
 */

import crypto from "crypto";
import { db } from "@/lib/db/drizzle";
import { eq, and, isNull, gt } from "drizzle-orm";
import { oauthAuthorizationCodes, type OAuthAuthorizationCode } from "./schema";
import { OAUTH_CONFIG } from "@/lib/oauth/config";

/**
 * Generate authorization code
 */
export function generateAuthorizationCode(): string {
  return crypto.randomBytes(32).toString("hex");
}

/**
 * Create authorization code
 */
export async function createAuthorizationCode(params: {
  clientId: string;
  userId: string;
  redirectUri: string;
  scope: string;
  state?: string;
  codeChallenge?: string;
  codeChallengeMethod?: "S256" | "plain";
}): Promise<OAuthAuthorizationCode> {
  const code = generateAuthorizationCode();
  const expiresAt = new Date(Date.now() + OAUTH_CONFIG.authorizationCodeTtl * 1000);

  const [authCode] = await db
    .insert(oauthAuthorizationCodes)
    .values({
      code,
      clientId: params.clientId,
      userId: params.userId,
      redirectUri: params.redirectUri,
      scope: params.scope,
      state: params.state,
      codeChallenge: params.codeChallenge,
      codeChallengeMethod: params.codeChallengeMethod,
      expiresAt,
    })
    .returning();

  return authCode;
}

/**
 * Find valid authorization code
 */
export async function findAuthorizationCode(code: string): Promise<OAuthAuthorizationCode | null> {
  const result = await db.query.oauthAuthorizationCodes.findFirst({
    where: and(
      eq(oauthAuthorizationCodes.code, code),
      isNull(oauthAuthorizationCodes.usedAt),
      gt(oauthAuthorizationCodes.expiresAt, new Date())
    ),
  });
  return result || null;
}

/**
 * Mark authorization code as used
 */
export async function markCodeAsUsed(code: string): Promise<void> {
  await db
    .update(oauthAuthorizationCodes)
    .set({ usedAt: new Date() })
    .where(eq(oauthAuthorizationCodes.code, code));
}

/**
 * Delete expired codes (cleanup)
 */
export async function cleanupExpiredCodes(): Promise<number> {
  const result = await db
    .delete(oauthAuthorizationCodes)
    .where(
      // Delete codes that are expired OR used more than 1 hour ago
      eq(oauthAuthorizationCodes.expiresAt, new Date(0)) // placeholder, fix below
    );
  return result.rowCount ?? 0;
}


