import { NextRequest } from "next/server";
import { db } from "@/lib/db/drizzle";
import { eq, and, isNotNull, desc } from "drizzle-orm";
import { getAppById } from "@/lib/models/oauth-clients/operations";
import { oauthAccessTokens, oauthRefreshTokens } from "@/lib/db/schemas";
import { users } from "@/lib/models/users/schema";

interface RouteContext {
  params: Promise<{ appId: string }>;
}

/**
 * Get all users who have authorized this app
 * A user is considered "authorized" if they have any token (active or not) for this client
 */
export async function GET(request: NextRequest, context: RouteContext) {
  const { appId } = await context.params;

  try {
    const app = await getAppById(appId);

    if (!app) {
      return Response.json({ error: "App not found" }, { status: 404 });
    }

    // Get all unique users who have tokens for this client
    const authorizedTokens = await db
      .selectDistinctOn([oauthAccessTokens.userId], {
        userId: oauthAccessTokens.userId,
        authorizedAt: oauthAccessTokens.createdAt,
        lastActivity: oauthAccessTokens.createdAt,
      })
      .from(oauthAccessTokens)
      .where(
        and(
          eq(oauthAccessTokens.clientId, app.clientId),
          isNotNull(oauthAccessTokens.userId)
        )
      )
      .orderBy(oauthAccessTokens.userId, desc(oauthAccessTokens.createdAt));

    // Get user details
    const userIds = authorizedTokens.map((t) => t.userId).filter(Boolean) as string[];

    if (userIds.length === 0) {
      return Response.json({ users: [] });
    }

    const userDetails = await db
      .select({
        id: users.id,
        firstName: users.firstName,
        lastName: users.lastName,
        email: users.email,
      })
      .from(users)
      .where(
        users.id.in ? users.id.in(userIds) : eq(users.id, userIds[0])
      );

    // Combine token data with user data
    const authorizedUsers = authorizedTokens
      .filter((t) => t.userId)
      .map((token) => {
        const user = userDetails.find((u) => u.id === token.userId);
        if (!user) return null;

        return {
          id: user.id,
          firstName: user.firstName,
          lastName: user.lastName,
          email: user.email,
          authorizedAt: token.authorizedAt,
          lastActivity: token.lastActivity,
        };
      })
      .filter(Boolean);

    return Response.json({ users: authorizedUsers });
  } catch (error) {
    console.error("Failed to fetch authorized users:", error);
    return Response.json({ error: "Failed to fetch users" }, { status: 500 });
  }
}


