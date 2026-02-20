import { NextRequest, NextResponse } from "next/server";
import { stripe } from "@/lib/stripe";
import { db } from "@/lib/db/drizzle";
import { spaces } from "@/lib/models/spaces/schema";
import { eq } from "drizzle-orm";

/**
 * Stripe redirects here after an organiser completes (or abandons) Connect onboarding.
 * We verify whether charges are enabled and update the space record accordingly.
 */
export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const spaceId = searchParams.get("spaceId");
  const appUrl = process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000";

  if (!spaceId) {
    return NextResponse.redirect(`${appUrl}/spaces`);
  }

  const space = await db.query.spaces.findFirst({
    where: eq(spaces.id, spaceId),
  });

  if (space?.stripeAccountId) {
    const account = await stripe.accounts.retrieve(space.stripeAccountId);

    if (account.charges_enabled) {
      await db
        .update(spaces)
        .set({ stripeAccountEnabled: true })
        .where(eq(spaces.id, spaceId));
    }
  }

  return NextResponse.redirect(
    `${appUrl}/spaces/${space?.slug}?tab=settings&stripeReturn=1`,
  );
}
