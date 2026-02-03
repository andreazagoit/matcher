/**
 * Get OAuth Client Info
 * GET /api/oauth/client-info?client_id=xxx
 */

import { NextRequest } from "next/server";
import { getClientByClientId } from "@/lib/models/oauth-clients/operations";

export async function GET(request: NextRequest) {
  const clientId = request.nextUrl.searchParams.get("client_id");

  if (!clientId) {
    return Response.json({ error: "client_id required" }, { status: 400 });
  }

  const client = await getClientByClientId(clientId);

  if (!client || !client.isActive) {
    return Response.json({ error: "Client not found" }, { status: 404 });
  }

  // Return public info only
  // Note: scopes are not configured per-client anymore
  return Response.json({
    name: client.name,
    description: client.description,
  });
}

