import { NextRequest } from "next/server";
import { db } from "@/lib/db/drizzle";
import { eq, and } from "drizzle-orm";
import { getAppById } from "@/lib/models/apps/operations";
import { accessTokens, refreshTokens } from "@/lib/db/schemas";

interface RouteContext {
  params: Promise<{ appId: string; userId: string }>;
}

/**
 * Revoke all tokens for a specific user on this app
 */
export async function POST(request: NextRequest, context: RouteContext) {
  const { appId, userId } = await context.params;

  try {
    const app = await getAppById(appId);

    if (!app) {
      return Response.json({ error: "App not found" }, { status: 404 });
    }

    const now = new Date();

    // Revoke all access tokens for this user
    await db
      .update(accessTokens)
      .set({ revokedAt: now })
      .where(
        and(
          eq(accessTokens.clientId, app.clientId),
          eq(accessTokens.userId, userId)
        )
      );

    // Revoke all refresh tokens for this user
    await db
      .update(refreshTokens)
      .set({ revokedAt: now })
      .where(
        and(
          eq(refreshTokens.clientId, app.clientId),
          eq(refreshTokens.userId, userId)
        )
      );

    return Response.json({ success: true });
  } catch (error) {
    console.error("Failed to revoke user access:", error);
    return Response.json({ error: "Failed to revoke access" }, { status: 500 });
  }
}


