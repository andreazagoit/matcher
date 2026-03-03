import { NextRequest, NextResponse } from "next/server";
import { headers } from "next/headers";
import { auth } from "@/lib/auth";
import { isValidCategory } from "@/lib/models/categories/data";
import { recordImpression } from "@/lib/models/impressions/operations";

/**
 * POST /api/users/interests
 * Save initial category interests for the authenticated user as impressions.
 * Called right after email OTP verification during sign-up onboarding.
 *
 * Each selected category is written as impression(action: 'liked') so the
 * ML graph can use them as user→category edges with a calibrated weight.
 */
export async function POST(req: NextRequest) {
  const session = await auth.api.getSession({ headers: await headers() });
  if (!session?.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json().catch(() => ({}));
  const categories = (body.categories as string[] | undefined)?.filter(
    (c) => isValidCategory(c),
  );

  if (!categories?.length) {
    return NextResponse.json({ ok: true });
  }

  for (const categoryId of categories) {
    recordImpression(session.user.id, categoryId, "category", "liked");
  }

  return NextResponse.json({ ok: true });
}
