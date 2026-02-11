/**
 * OAuth Authorization Code Generation
 * POST /oauth/api/authorize
 * 
 * Called after user login and consent
 */

import { NextRequest } from "next/server";
import { cookies } from "next/headers";
import { OAuthError, OAuthErrors } from "@/lib/oauth/errors";
import { validateAuthorizeRequest, createAuthCode } from "@/lib/oauth/grants/authorization-code";
import { getUserById } from "@/lib/models/users/operations";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const {
      client_id: clientId,
      redirect_uri: redirectUri,
      scope,
      state,
      code_challenge: codeChallenge,
      code_challenge_method: codeChallengeMethod,
    } = body;

    // Validate request
    await validateAuthorizeRequest({
      responseType: "code",
      clientId,
      redirectUri,
      scope,
      state,
      codeChallenge,
      codeChallengeMethod,
    });

    // Get user ID from session (set by login)
    const cookieStore = await cookies();
    const userId = cookieStore.get("user_id")?.value;

    if (!userId) {
      throw OAuthErrors.accessDenied("User not authenticated");
    }

    // Verify user actually exists in current DB (prevents ghost sessions after DB reset)
    const userExists = await getUserById(userId);
    if (!userExists) {
      // Clear invalid cookie
      cookieStore.delete("user_id");
      throw OAuthErrors.accessDenied("Session invalid: User no longer exists");
    }

    // Create authorization code
    const code = await createAuthCode({
      clientId,
      userId,
      redirectUri,
      scope: scope || "",
      state,
      codeChallenge,
      codeChallengeMethod,
    });

    // Build redirect URL
    const redirectUrl = new URL(redirectUri);
    redirectUrl.searchParams.set("code", code);
    if (state) {
      redirectUrl.searchParams.set("state", state);
    }

    return Response.json({ redirect_url: redirectUrl.toString() });
  } catch (error) {
    if (error instanceof OAuthError) {
      return Response.json(error.toJSON(), { status: error.statusCode });
    }

    console.error("OAuth authorize error:", error);
    return Response.json(
      OAuthErrors.serverError("Internal server error").toJSON(),
      { status: 500 }
    );
  }
}


