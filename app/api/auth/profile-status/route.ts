/**
 * Profile Status API
 * GET /api/auth/profile-status
 * 
 * Check if user has completed their profile/assessment
 */

import { cookies } from "next/headers";
import { hasCompleteProfile } from "@/lib/models/profiles/operations";
import { getUserById } from "@/lib/models/users/operations";
import { auth } from "@/lib/auth";

export async function GET() {
  try {
    const session = await auth();
    const cookieStore = await cookies();
    const userId = session?.user?.id || cookieStore.get("user_id")?.value;

    if (!userId) {
      return Response.json(
        { authenticated: false, hasProfile: false },
        { status: 200 }
      );
    }

    const user = await getUserById(userId);
    if (!user) {
      return Response.json(
        { authenticated: false, hasProfile: false },
        { status: 200 }
      );
    }

    const hasProfile = await hasCompleteProfile(userId);

    return Response.json({
      authenticated: true,
      hasProfile,
      user: {
        id: user.id,
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.email,
      },
    });
  } catch (error) {
    console.error("Profile status error:", error);
    return Response.json(
      { error: "Failed to check profile status" },
      { status: 500 }
    );
  }
}


