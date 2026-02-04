/**
 * Single App API
 * GET /api/dashboard/clients/[appId] - Get app details
 * PATCH /api/dashboard/clients/[appId] - Update app
 * DELETE /api/dashboard/clients/[appId] - Delete app
 */

import { NextRequest } from "next/server";
import { db } from "@/lib/db/drizzle";
import { eq, and, gt, count, isNull, isNotNull, countDistinct } from "drizzle-orm";
import {
  getAppById,
  updateApp,
  deleteApp,
} from "@/lib/models/apps/operations";
import { accessTokens, refreshTokens } from "@/lib/db/schemas";

interface RouteContext {
  params: Promise<{ appId: string }>;
}

export async function GET(request: NextRequest, context: RouteContext) {
  const { appId } = await context.params;

  try {
    const app = await getAppById(appId);

    if (!app) {
      return Response.json({ error: "App not found" }, { status: 404 });
    }

    // Get token stats
    const now = new Date();

    const [activeAccessResult] = await db
      .select({ count: count() })
      .from(accessTokens)
      .where(
        and(
          eq(accessTokens.clientId, app.clientId),
          gt(accessTokens.expiresAt, now),
          isNull(accessTokens.revokedAt)
        )
      );

    const [activeRefreshResult] = await db
      .select({ count: count() })
      .from(refreshTokens)
      .where(
        and(
          eq(refreshTokens.clientId, app.clientId),
          gt(refreshTokens.expiresAt, now),
          isNull(refreshTokens.revokedAt)
        )
      );

    const [totalAccessResult] = await db
      .select({ count: count() })
      .from(accessTokens)
      .where(eq(accessTokens.clientId, app.clientId));

    // Count unique authorized users (distinct userId with non-revoked tokens)
    const [authorizedUsersResult] = await db
      .select({ count: countDistinct(accessTokens.userId) })
      .from(accessTokens)
      .where(
        and(
          eq(accessTokens.clientId, app.clientId),
          isNotNull(accessTokens.userId),
          isNull(accessTokens.revokedAt)
        )
      );

    return Response.json({
      app: {
        id: app.id,
        name: app.name,
        description: app.description,
        clientId: app.clientId,
        secretKey: app.secretKey,
        redirectUris: app.redirectUris,
        accessTokenTtl: app.accessTokenTtl,
        refreshTokenTtl: app.refreshTokenTtl,
        isActive: app.isActive,
        createdAt: app.createdAt,
        updatedAt: app.updatedAt,
      },
      stats: {
        activeAccessTokens: activeAccessResult?.count || 0,
        activeRefreshTokens: activeRefreshResult?.count || 0,
        totalTokensIssued: totalAccessResult?.count || 0,
        authorizedUsersCount: authorizedUsersResult?.count || 0,
      },
    });
  } catch (error) {
    console.error("Failed to fetch app:", error);
    return Response.json({ error: "Failed to fetch app" }, { status: 500 });
  }
}

export async function PATCH(request: NextRequest, context: RouteContext) {
  const { appId } = await context.params;

  try {
    const body = await request.json();
    const { name, description, redirectUris, isActive } = body;

    const updated = await updateApp(appId, {
      name,
      description,
      redirectUris,
      isActive,
    });

    if (!updated) {
      return Response.json({ error: "App not found" }, { status: 404 });
    }

    return Response.json({ app: updated });
  } catch (error) {
    console.error("Failed to update app:", error);
    return Response.json({ error: "Failed to update app" }, { status: 500 });
  }
}

export async function DELETE(request: NextRequest, context: RouteContext) {
  const { appId } = await context.params;

  try {
    const deleted = await deleteApp(appId);

    if (!deleted) {
      return Response.json({ error: "App not found" }, { status: 404 });
    }

    return Response.json({ success: true });
  } catch (error) {
    console.error("Failed to delete app:", error);
    return Response.json({ error: "Failed to delete app" }, { status: 500 });
  }
}
