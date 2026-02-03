/**
 * Dashboard Apps API
 * GET /api/dashboard/clients - List all apps
 * POST /api/dashboard/clients - Create new app
 */

import { NextRequest } from "next/server";
import {
  getAllApps,
  createOAuthApp,
} from "@/lib/models/oauth-clients/operations";

export async function GET() {
  try {
    const apps = await getAllApps();
    
    return Response.json({
      apps: apps.map((app) => ({
        id: app.id,
        name: app.name,
        description: app.description,
        clientId: app.clientId,
        redirectUris: app.redirectUris,
        isActive: app.isActive,
        createdAt: app.createdAt,
        updatedAt: app.updatedAt,
      })),
    });
  } catch (error) {
    console.error("Failed to fetch apps:", error);
    return Response.json({ error: "Failed to fetch apps" }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    const { name, description, redirectUris } = body;

    if (!name) {
      return Response.json(
        { error: "Name is required" },
        { status: 400 }
      );
    }

    const { app, clientId, secretKey } = await createOAuthApp({
      name,
      description,
      redirectUris,
    });

    return Response.json({
      app: {
        id: app.id,
        name: app.name,
        description: app.description,
        redirectUris: app.redirectUris,
        isActive: app.isActive,
        createdAt: app.createdAt,
      },
      credentials: {
        // For OAuth (public, can be in frontend)
        clientId,
        // For M2M (secret, only backend) - only shown once!
        secretKey,
      },
    });
  } catch (error) {
    console.error("Failed to create app:", error);
    return Response.json({ error: "Failed to create app" }, { status: 500 });
  }
}
