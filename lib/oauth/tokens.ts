/**
 * OAuth 2.0 Token Generation & Validation
 * JWT-based access tokens
 */

import crypto from "crypto";
import { db } from "@/lib/db/drizzle";
import { eq, and, isNull, gt } from "drizzle-orm";
import { oauthAccessTokens, oauthRefreshTokens } from "@/lib/models/oauth-tokens/schema";
import { OAUTH_CONFIG } from "./config";

// Simple JWT implementation (no external deps)
// In production, consider using jose or jsonwebtoken

interface JWTHeader {
  alg: "HS256";
  typ: "JWT";
}

interface AccessTokenPayload {
  iss: string;
  sub: string; // user_id or client_id
  aud: string; // client_id
  exp: number;
  iat: number;
  jti: string;
  scope: string;
  client_id: string;
  user_id?: string;
}

interface RefreshTokenPayload {
  jti: string;
  client_id: string;
  user_id?: string;
  scope: string;
}

const JWT_SECRET = process.env.JWT_SECRET || "change-me-in-production";

/**
 * Base64URL encode
 */
function base64url(data: string | Buffer): string {
  const buf = typeof data === "string" ? Buffer.from(data) : data;
  return buf.toString("base64url");
}

/**
 * Base64URL decode
 */
function base64urlDecode(data: string): string {
  return Buffer.from(data, "base64url").toString();
}

/**
 * Create HMAC SHA256 signature
 */
function sign(data: string, secret: string): string {
  return crypto.createHmac("sha256", secret).update(data).digest("base64url");
}

/**
 * Hash token for storage
 */
export function hashToken(token: string): string {
  return crypto.createHash("sha256").update(token).digest("hex");
}

/**
 * Generate a random token ID (jti)
 */
export function generateJti(): string {
  return crypto.randomBytes(16).toString("hex");
}

/**
 * Generate access token JWT
 */
export function generateAccessToken(params: {
  clientId: string;
  userId?: string;
  scope: string;
  ttl?: number;
}): { token: string; jti: string; expiresAt: Date } {
  const ttl = params.ttl || OAUTH_CONFIG.accessTokenTtl;
  const now = Math.floor(Date.now() / 1000);
  const jti = generateJti();

  const header: JWTHeader = { alg: "HS256", typ: "JWT" };

  const payload: AccessTokenPayload = {
    iss: OAUTH_CONFIG.issuer,
    sub: params.userId || params.clientId,
    aud: params.clientId,
    exp: now + ttl,
    iat: now,
    jti,
    scope: params.scope,
    client_id: params.clientId,
    ...(params.userId && { user_id: params.userId }),
  };

  const headerB64 = base64url(JSON.stringify(header));
  const payloadB64 = base64url(JSON.stringify(payload));
  const signature = sign(`${headerB64}.${payloadB64}`, JWT_SECRET);

  const token = `${headerB64}.${payloadB64}.${signature}`;
  const expiresAt = new Date((now + ttl) * 1000);

  return { token, jti, expiresAt };
}

/**
 * Generate refresh token (opaque)
 */
export function generateRefreshToken(params: {
  clientId: string;
  userId?: string;
  scope: string;
  ttl?: number;
}): { token: string; jti: string; expiresAt: Date } {
  const ttl = params.ttl || OAUTH_CONFIG.refreshTokenTtl;
  const jti = generateJti();

  // Opaque token with embedded data
  const payload: RefreshTokenPayload = {
    jti,
    client_id: params.clientId,
    user_id: params.userId,
    scope: params.scope,
  };

  const token = `rt_${base64url(JSON.stringify(payload))}_${crypto.randomBytes(16).toString("hex")}`;
  const expiresAt = new Date(Date.now() + ttl * 1000);

  return { token, jti, expiresAt };
}

/**
 * Verify and decode access token
 */
export function verifyAccessToken(token: string): AccessTokenPayload | null {
  try {
    const parts = token.split(".");
    if (parts.length !== 3) return null;

    const [headerB64, payloadB64, signature] = parts;

    // Verify signature (timing-safe comparison)
    const expectedSignature = sign(`${headerB64}.${payloadB64}`, JWT_SECRET);
    const sigBuffer = Buffer.from(signature);
    const expectedBuffer = Buffer.from(expectedSignature);
    if (sigBuffer.length !== expectedBuffer.length ||
      !crypto.timingSafeEqual(sigBuffer, expectedBuffer)) {
      return null;
    }

    // Decode payload
    const payload: AccessTokenPayload = JSON.parse(base64urlDecode(payloadB64));

    // Check expiration
    if (payload.exp < Math.floor(Date.now() / 1000)) return null;

    return payload;
  } catch {
    return null;
  }
}

/**
 * Store access token in database
 */
export async function storeAccessToken(params: {
  tokenHash: string;
  jti: string;
  clientId: string;
  userId?: string;
  scope: string;
  expiresAt: Date;
}) {
  const [token] = await db
    .insert(oauthAccessTokens)
    .values({
      tokenHash: params.tokenHash,
      jti: params.jti,
      clientId: params.clientId,
      userId: params.userId,
      scope: params.scope,
      expiresAt: params.expiresAt,
    })
    .returning();

  return token;
}

/**
 * Store refresh token in database
 */
export async function storeRefreshToken(params: {
  tokenHash: string;
  jti: string;
  accessTokenId: string;
  clientId: string;
  userId?: string;
  scope: string;
  expiresAt: Date;
}) {
  const [token] = await db
    .insert(oauthRefreshTokens)
    .values({
      tokenHash: params.tokenHash,
      jti: params.jti,
      accessTokenId: params.accessTokenId,
      clientId: params.clientId,
      userId: params.userId,
      scope: params.scope,
      expiresAt: params.expiresAt,
    })
    .returning();

  return token;
}

/**
 * Find valid access token by jti
 */
export async function findAccessTokenByJti(jti: string) {
  const result = await db.query.oauthAccessTokens.findFirst({
    where: and(
      eq(oauthAccessTokens.jti, jti),
      isNull(oauthAccessTokens.revokedAt),
      gt(oauthAccessTokens.expiresAt, new Date())
    ),
  });
  return result;
}

/**
 * Find valid refresh token by hash
 */
export async function findRefreshTokenByHash(hash: string) {
  const result = await db.query.oauthRefreshTokens.findFirst({
    where: and(
      eq(oauthRefreshTokens.tokenHash, hash),
      isNull(oauthRefreshTokens.revokedAt),
      gt(oauthRefreshTokens.expiresAt, new Date())
    ),
  });
  return result;
}

/**
 * Revoke access token
 */
export async function revokeAccessToken(jti: string) {
  await db
    .update(oauthAccessTokens)
    .set({ revokedAt: new Date() })
    .where(eq(oauthAccessTokens.jti, jti));
}

/**
 * Revoke refresh token and associated access token
 */
export async function revokeRefreshToken(jti: string) {
  const refreshToken = await db.query.oauthRefreshTokens.findFirst({
    where: eq(oauthRefreshTokens.jti, jti),
  });

  if (refreshToken) {
    // Revoke refresh token
    await db
      .update(oauthRefreshTokens)
      .set({ revokedAt: new Date() })
      .where(eq(oauthRefreshTokens.jti, jti));

    // Revoke associated access token
    if (refreshToken.accessTokenId) {
      await db
        .update(oauthAccessTokens)
        .set({ revokedAt: new Date() })
        .where(eq(oauthAccessTokens.id, refreshToken.accessTokenId));
    }
  }
}


