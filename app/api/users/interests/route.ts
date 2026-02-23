import { NextRequest, NextResponse } from "next/server";
import { headers } from "next/headers";
import { auth } from "@/lib/auth";
import { db } from "@/lib/db/drizzle";
import { userInterests } from "@/lib/models/interests/schema";

/**
 * POST /api/users/interests
 * Save initial interests for the authenticated user.
 * Called right after email OTP verification during sign-up.
 */
export async function POST(req: NextRequest) {
  const session = await auth.api.getSession({ headers: await headers() });
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json().catch(() => ({}));
  const tags = body.tags as string[] | undefined;

  if (!tags?.length) {
    return NextResponse.json({ ok: true });
  }

  await db
    .insert(userInterests)
    .values(tags.map((tag: string) => ({ userId: session.user.id, tag })))
    .onConflictDoNothing();

  return NextResponse.json({ ok: true });
}
