/**
 * Rotate Secret Key API
 * POST /api/dashboard/clients/[appId]/rotate-secret
 */

import { NextRequest } from "next/server";
import { rotateSecretKey } from "@/lib/models/apps/operations";

interface RouteContext {
  params: Promise<{ appId: string }>;
}

export async function POST(request: NextRequest, context: RouteContext) {
  const { appId } = await context.params;

  try {
    const result = await rotateSecretKey(appId);

    if (!result) {
      return Response.json(
        { error: "App not found" },
        { status: 404 }
      );
    }

    return Response.json({
      success: true,
      // Only shown once!
      secretKey: result.secretKey,
    });
  } catch (error) {
    console.error("Failed to rotate secret:", error);
    return Response.json(
      { error: "Failed to rotate secret" },
      { status: 500 }
    );
  }
}
