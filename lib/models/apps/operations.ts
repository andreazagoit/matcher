/**
 * OAuth Apps Operations
 * 
 * Hybrid system:
 * - OAuth (client_id) for user authorization flows
 * - API Key (secret_key) for direct M2M access
 */

import crypto from "crypto";
import { db } from "@/lib/db/drizzle";
import { eq } from "drizzle-orm";
import { apps, type App } from "./schema";

// ============================================
// CREDENTIAL GENERATION
// ============================================

export function generateClientId(): string {
  return `client_${crypto.randomBytes(16).toString("hex")}`;
}

export function generateSecretKey(): string {
  return `sk_live_${crypto.randomBytes(32).toString("hex")}`;
}

export function hashSecret(secret: string): string {
  return crypto.createHash("sha256").update(secret).digest("hex");
}

export function verifySecret(secret: string, hash: string): boolean {
  return hashSecret(secret) === hash;
}

// ============================================
// APP CRUD OPERATIONS
// ============================================

export interface CreateAppResult {
  app: App;
  /** OAuth client_id (can be public) */
  clientId: string;
  /** M2M secret key */
  secretKey: string;
}

/**
 * Create a new OAuth App
 * Returns both client_id (for OAuth) and secret_key (for M2M)
 */
export async function createApp(params: {
  name: string;
  description?: string;
  redirectUris?: string[];
  ownerId: string;
}): Promise<CreateAppResult> {
  const clientId = generateClientId();
  const secretKey = generateSecretKey();
  const secretKeyHash = hashSecret(secretKey);

  const [app] = await db
    .insert(apps)
    .values({
      name: params.name,
      description: params.description,
      clientId,
      secretKey,
      secretKeyHash,
      redirectUris: params.redirectUris || [],
      ownerId: params.ownerId,
    })
    .returning();

  return {
    app,
    clientId,
    secretKey,
  };
}

/**
 * Get app by ID
 */
export async function getAppById(id: string): Promise<App | null> {
  const result = await db.query.apps.findFirst({
    where: eq(apps.id, id),
  });
  return result || null;
}

/**
 * Get app by client_id (for OAuth flows)
 */
export async function getAppByClientId(clientId: string): Promise<App | null> {
  const result = await db.query.apps.findFirst({
    where: eq(apps.clientId, clientId),
  });
  return result || null;
}

/**
 * Get all apps
 */
export async function getAllApps(): Promise<App[]> {
  return db.query.apps.findMany({
    orderBy: (apps, { desc }) => [desc(apps.createdAt)],
  });
}

/**
 * Get apps by owner
 */
export async function getAppsByOwner(ownerId: string): Promise<App[]> {
  return db.query.apps.findMany({
    where: eq(apps.ownerId, ownerId),
    orderBy: (apps, { desc }) => [desc(apps.createdAt)],
  });
}

/**
 * Update app
 */
export async function updateApp(
  id: string,
  data: Partial<Pick<App, "name" | "description" | "redirectUris" | "isActive">>
): Promise<App | null> {
  const [updated] = await db
    .update(apps)
    .set({ ...data, updatedAt: new Date() })
    .where(eq(apps.id, id))
    .returning();
  return updated || null;
}

/**
 * Delete app
 */
export async function deleteApp(id: string): Promise<boolean> {
  const result = await db.delete(apps).where(eq(apps.id, id)).returning();
  return result.length > 0;
}

/**
 * Rotate secret key
 */
export async function rotateSecretKey(id: string): Promise<{ app: App; secretKey: string } | null> {
  const app = await getAppById(id);
  if (!app) return null;

  const newSecretKey = generateSecretKey();
  const newSecretKeyHash = hashSecret(newSecretKey);

  const [updated] = await db
    .update(apps)
    .set({
      secretKey: newSecretKey,
      secretKeyHash: newSecretKeyHash,
      updatedAt: new Date()
    })
    .where(eq(apps.id, id))
    .returning();

  return updated ? { app: updated, secretKey: newSecretKey } : null;
}

// ============================================
// AUTHENTICATION
// ============================================

/**
 * Validate M2M API Key (direct access)
 * Used for: Authorization: Bearer sk_live_xxx
 */
export async function validateApiKey(secretKey: string): Promise<App | null> {
  // Get all active apps and check the hash
  // Note: In production with many apps, you'd want to index or cache this
  const appsList = await db.query.apps.findMany({
    where: eq(apps.isActive, true),
  });

  for (const app of appsList) {
    if (verifySecret(secretKey, app.secretKeyHash)) {
      return app;
    }
  }

  return null;
}

/**
 * Validate OAuth client (for authorization_code flow)
 * client_id is public, no secret needed for the authorize step
 */
export async function validateOAuthClient(clientId: string): Promise<App | null> {
  const app = await getAppByClientId(clientId);
  if (!app || !app.isActive) return null;
  return app;
}

/**
 * Check if redirect URI is allowed
 */
export function isRedirectUriAllowed(app: App | null, redirectUri: string): boolean {
  if (!app) return false;
  return app.redirectUris?.includes(redirectUri) ?? false;
}

export async function validateClientCredentials(
  clientId: string,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _clientSecret?: string
): Promise<App | null> {
  // For OAuth authorization_code, we only validate client_id
  // The secret_key is for M2M, not OAuth
  return validateOAuthClient(clientId);
}

export function clientSupportsGrant(_app: App, grantType: string): boolean {
  // All apps support these grant types
  const supportedGrants = ["authorization_code", "refresh_token"];
  return supportedGrants.includes(grantType);
}
