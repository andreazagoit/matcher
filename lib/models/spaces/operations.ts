/**
 * Spaces Operations
 * 
 * Hybrid system:
 * - OAuth (client_id) for user authorization flows
 * - API Key (secret_key) for direct M2M access
 */

import crypto from "crypto";
import { db } from "@/lib/db/drizzle";
import { spaces, type Space } from "./schema";
import { members } from "@/lib/models/members/schema";
import { eq } from "drizzle-orm";

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

export function generateSlug(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

// ============================================
// SPACE CRUD OPERATIONS
// ============================================

export interface CreateSpaceResult {
  space: Space;
  clientId: string;
  secretKey: string;
}

export async function createSpace(params: {
  name: string;
  slug?: string;
  description?: string;
  redirectUris?: string[];
  creatorId: string; // The user who creates the space becomes an admin
  visibility?: "public" | "private" | "hidden";
  joinPolicy?: "open" | "apply" | "invite_only";
}): Promise<CreateSpaceResult> {
  const clientId = generateClientId();
  const secretKey = generateSecretKey();
  const secretKeyHash = hashSecret(secretKey);
  const slug = params.slug || generateSlug(params.name);

  return await db.transaction(async (tx) => {
    const [space] = await tx
      .insert(spaces)
      .values({
        name: params.name,
        slug,
        description: params.description,
        clientId,
        secretKey,
        secretKeyHash,
        redirectUris: params.redirectUris || [],
        visibility: params.visibility || "public",
        joinPolicy: params.joinPolicy || "open",
      })
      .returning();

    // Create the creator as admin
    await tx.insert(members).values({
      spaceId: space.id,
      userId: params.creatorId,
      role: "admin",
      status: "active",
    });

    return {
      space,
      clientId,
      secretKey,
    };
  });
}

export async function getSpaceById(id: string): Promise<Space | null> {
  const result = await db.query.spaces.findFirst({
    where: eq(spaces.id, id),
  });
  return result || null;
}

export async function getSpaceBySlug(slug: string): Promise<Space | null> {
  const result = await db.query.spaces.findFirst({
    where: eq(spaces.slug, slug),
  });
  return result || null;
}

export async function getSpaceByClientId(clientId: string): Promise<Space | null> {
  const result = await db.query.spaces.findFirst({
    where: eq(spaces.clientId, clientId),
  });
  return result || null;
}

export async function getAllSpaces(): Promise<Space[]> {
  return db.query.spaces.findMany({
    orderBy: (spaces, { desc }) => [desc(spaces.createdAt)],
  });
}

/**
 * Update space
 */
export async function updateSpace(
  id: string,
  data: Partial<Pick<Space, "name" | "slug" | "description" | "redirectUris" | "isActive" | "visibility" | "logoUrl" | "joinPolicy">>
): Promise<Space | null> {
  const [updated] = await db
    .update(spaces)
    .set({ ...data, updatedAt: new Date() })
    .where(eq(spaces.id, id))
    .returning();
  return updated || null;
}

/**
 * Delete space
 */
export async function deleteSpace(id: string): Promise<boolean> {
  const result = await db.delete(spaces).where(eq(spaces.id, id)).returning();
  return result.length > 0;
}

/**
 * Rotate secret key
 */
export async function rotateSecretKey(id: string): Promise<{ space: Space; secretKey: string } | null> {
  const space = await getSpaceById(id);
  if (!space) return null;

  const newSecretKey = generateSecretKey();
  const newSecretKeyHash = hashSecret(newSecretKey);

  const [updated] = await db
    .update(spaces)
    .set({
      secretKey: newSecretKey,
      secretKeyHash: newSecretKeyHash,
      updatedAt: new Date()
    })
    .where(eq(spaces.id, id))
    .returning();

  return updated ? { space: updated, secretKey: newSecretKey } : null;
}

// ============================================
// AUTHENTICATION
// ============================================

export async function validateApiKey(secretKey: string): Promise<Space | null> {
  const spacesList = await db.query.spaces.findMany({
    where: eq(spaces.isActive, true),
  });

  for (const space of spacesList) {
    if (verifySecret(secretKey, space.secretKeyHash)) {
      return space;
    }
  }

  return null;
}

export async function validateOAuthClient(clientId: string): Promise<Space | null> {
  const space = await getSpaceByClientId(clientId);
  if (!space || !space.isActive) return null;
  return space;
}

export function isRedirectUriAllowed(space: Space | null, redirectUri: string): boolean {
  if (!space) return false;
  return space.redirectUris?.includes(redirectUri) ?? false;
}

export async function validateClientCredentials(
  clientId: string,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _clientSecret?: string
): Promise<Space | null> {
  return validateOAuthClient(clientId);
}

export function clientSupportsGrant(_space: Space, grantType: string): boolean {
  const supportedGrants = ["authorization_code", "refresh_token"];
  return supportedGrants.includes(grantType);
}
