/**
 * Complete Assessment API
 * POST /api/complete-assessment
 * 
 * Saves assessment answers and generates user profile with embeddings
 */

import { NextRequest } from "next/server";
import { cookies } from "next/headers";
import { auth } from "@/lib/oauth/auth";
import { db } from "@/lib/db/drizzle";
import { assessments } from "@/lib/models/assessments/schema";
import { ASSESSMENT_NAME } from "@/lib/models/assessments/questions";
import { assembleProfile } from "@/lib/models/assessments/assembler";
import { upsertProfile } from "@/lib/models/profiles/operations";

export async function POST(request: NextRequest) {
  try {
    // Try next-auth session first, fall back to user_id cookie (for OAuth flow)
    const session = await auth();
    const cookieStore = await cookies();
    const userId = session?.user?.id || cookieStore.get("user_id")?.value;

    if (!userId) {
      return Response.json(
        { error: "Not authenticated" },
        { status: 401 }
      );
    }

    const body = await request.json();
    const { answers } = body;

    if (!answers || typeof answers !== "object") {
      return Response.json(
        { error: "Answers are required" },
        { status: 400 }
      );
    }

    // 1. Save assessment
    const [assessment] = await db
      .insert(assessments)
      .values({
        userId,
        assessmentName: ASSESSMENT_NAME,
        status: "completed",
        answers,
        completedAt: new Date(),
      })
      .returning();

    // 2. Assemble profile data from answers
    const profileData = assembleProfile(answers);

    // 3. Generate embeddings and save profile
    const profile = await upsertProfile(userId, profileData, 1);

    return Response.json({
      success: true,
      assessmentId: assessment.id,
      profileId: profile.id,
    });
  } catch (error) {
    console.error("Complete assessment error:", error);
    return Response.json(
      { error: error instanceof Error ? error.message : "Failed to complete assessment" },
      { status: 500 }
    );
  }
}
