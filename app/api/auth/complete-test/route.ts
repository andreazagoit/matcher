/**
 * Complete Test API
 * POST /api/auth/complete-test
 * 
 * Saves test answers and generates user profile with embeddings
 */

import { NextRequest } from "next/server";
import { cookies } from "next/headers";
import { db } from "@/lib/db/drizzle";
import { testSessions } from "@/lib/models/tests/schema";
import { TEST_NAME } from "@/lib/models/tests/questions";
import { assembleProfile } from "@/lib/models/tests/assembler";
import { upsertUserProfile } from "@/lib/models/profiles/operations";

export async function POST(request: NextRequest) {
  try {
    const cookieStore = await cookies();
    const userId = cookieStore.get("user_id")?.value;

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

    // 1. Save test session
    const [testSession] = await db
      .insert(testSessions)
      .values({
        userId,
        testName: TEST_NAME,
        version: "1.0",
        status: "completed",
        progress: 100,
        answers,
        completedAt: new Date(),
      })
      .returning();

    // 2. Assemble profile data from answers
    const profileData = assembleProfile(answers);

    // 3. Generate embeddings and save profile
    const profile = await upsertUserProfile(userId, profileData, 1);

    return Response.json({
      success: true,
      testSessionId: testSession.id,
      profileId: profile.id,
    });
  } catch (error) {
    console.error("Complete test error:", error);
    return Response.json(
      { error: error instanceof Error ? error.message : "Failed to complete test" },
      { status: 500 }
    );
  }
}

