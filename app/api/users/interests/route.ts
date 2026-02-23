import { NextRequest, NextResponse } from "next/server";
import { headers } from "next/headers";
import { auth } from "@/lib/auth";
import { db } from "@/lib/db/drizzle";
import { users } from "@/lib/models/users/schema";
import { eq } from "drizzle-orm";
import { isValidTag } from "@/lib/models/tags/data";

/**
 * POST /api/users/interests
 * Save initial tags for the authenticated user.
 * Called right after email OTP verification during sign-up.
 */
export async function POST(req: NextRequest) {
  const session = await auth.api.getSession({ headers: await headers() });
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json().catch(() => ({}));
  const tags = (body.tags as string[] | undefined)?.filter((t) => isValidTag(t));

  if (!tags?.length) {
    return NextResponse.json({ ok: true });
  }

  await db
    .update(users)
    .set({ tags, updatedAt: new Date() })
    .where(eq(users.id, session.user.id));

  return NextResponse.json({ ok: true });
}
