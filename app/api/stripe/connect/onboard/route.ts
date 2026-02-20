import { NextRequest, NextResponse } from "next/server";
import { getAuthContext } from "@/lib/auth/utils";
import { stripe } from "@/lib/stripe";
import { db } from "@/lib/db/drizzle";
import { spaces } from "@/lib/models/spaces/schema";
import { eq } from "drizzle-orm";

export async function POST(req: NextRequest) {
  const authContext = await getAuthContext();
  if (!authContext.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { spaceId } = await req.json();
  if (!spaceId) {
    return NextResponse.json({ error: "spaceId is required" }, { status: 400 });
  }

  const space = await db.query.spaces.findFirst({
    where: eq(spaces.id, spaceId),
  });

  if (!space) {
    return NextResponse.json({ error: "Space not found" }, { status: 404 });
  }
  if (space.ownerId !== authContext.user.id) {
    return NextResponse.json({ error: "Only the space owner can connect Stripe" }, { status: 403 });
  }

  let stripeAccountId = space.stripeAccountId;

  const appUrl = process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000";

  try {
    // If a connected account already exists, decide whether to send the user
    // to onboarding (incomplete) or to the Stripe Express Dashboard (complete).
    if (stripeAccountId) {
      const existingAccount = await stripe.accounts.retrieve(stripeAccountId);
      const isEnabled = !!existingAccount.charges_enabled;

      // Keep local DB in sync with Stripe's source of truth.
      if (space.stripeAccountEnabled !== isEnabled) {
        await db
          .update(spaces)
          .set({ stripeAccountEnabled: isEnabled })
          .where(eq(spaces.id, spaceId));
      }

      if (isEnabled) {
        const loginLink = await stripe.accounts.createLoginLink(stripeAccountId);
        return NextResponse.json({ url: loginLink.url });
      }
    }

    // Create a new Express account if one doesn't exist yet
    if (!stripeAccountId) {
      const account = await stripe.accounts.create({
        type: "express",
        capabilities: {
          card_payments: { requested: true },
          transfers: { requested: true },
        },
        metadata: { spaceId, spaceName: space.name },
      });

      stripeAccountId = account.id;

      await db
        .update(spaces)
        .set({ stripeAccountId })
        .where(eq(spaces.id, spaceId));
    }

    const settingsUrl = `${appUrl}/spaces/${space.slug}?tab=settings`;
    const accountLink = await stripe.accountLinks.create({
      account: stripeAccountId,
      // If the link expires, send the user back to settings and let them click again.
      refresh_url: `${settingsUrl}&stripeRefresh=1`,
      return_url: `${appUrl}/api/stripe/connect/return?spaceId=${spaceId}`,
      type: "account_onboarding",
    });

    return NextResponse.json({ url: accountLink.url });
  } catch (err: unknown) {
    const stripeErr = err as { statusCode?: number; message?: string; requestId?: string };
    if (stripeErr?.statusCode === 400 && stripeErr?.message?.includes("Connect")) {
      return NextResponse.json(
        {
          error:
            "Stripe Connect non è abilitato su questo account. Abilitalo dal dashboard Stripe (Settings > Connect).",
          requestId: stripeErr.requestId ?? null,
        },
        { status: 400 },
      );
    }
    console.error("Stripe Connect onboard error:", err);
    return NextResponse.json({ error: "Errore interno del server" }, { status: 500 });
  }
}
