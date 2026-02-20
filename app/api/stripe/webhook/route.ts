import { NextRequest, NextResponse } from "next/server";
import { stripe } from "@/lib/stripe";
import { db } from "@/lib/db/drizzle";
import { eventAttendees } from "@/lib/models/events/schema";
import { spaces } from "@/lib/models/spaces/schema";
import { eq, and } from "drizzle-orm";
import type Stripe from "stripe";

// Disable body parsing so we can verify the raw Stripe signature
export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  const rawBody = await req.text();
  const sig = req.headers.get("stripe-signature");
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;

  if (!sig) {
    return NextResponse.json({ error: "Missing stripe-signature" }, { status: 400 });
  }
  if (!webhookSecret) {
    return NextResponse.json(
      {
        error:
          "Missing STRIPE_WEBHOOK_SECRET. Set STRIPE_WEBHOOK_SECRET=whsec_... in .env and restart the server.",
      },
      { status: 500 },
    );
  }

  let event: Stripe.Event;

  try {
    event = stripe.webhooks.constructEvent(
      rawBody,
      sig,
      webhookSecret,
    );
  } catch (err) {
    const message = err instanceof Error ? err.message : "Invalid signature";
    return NextResponse.json({ error: message }, { status: 400 });
  }

  switch (event.type) {
    case "checkout.session.completed": {
      const session = event.data.object as Stripe.Checkout.Session;
      const checkoutSessionId = session.id;

      // Mark the attendee record as paid
      await db
        .update(eventAttendees)
        .set({ paymentStatus: "paid", status: "going" })
        .where(eq(eventAttendees.stripeCheckoutSessionId, checkoutSessionId));

      break;
    }

    case "checkout.session.expired": {
      const session = event.data.object as Stripe.Checkout.Session;

      // Remove the pending attendee record so the user can try again
      await db
        .delete(eventAttendees)
        .where(
          and(
            eq(eventAttendees.stripeCheckoutSessionId, session.id),
            eq(eventAttendees.paymentStatus, "pending"),
          ),
        );

      break;
    }

    case "account.updated": {
      // Fired for connected accounts — update stripeAccountEnabled based on charges_enabled
      const account = event.data.object as Stripe.Account;

      const space = await db.query.spaces.findFirst({
        where: eq(spaces.stripeAccountId, account.id),
        columns: { id: true },
      });

      if (space) {
        await db
          .update(spaces)
          .set({ stripeAccountEnabled: account.charges_enabled })
          .where(eq(spaces.id, space.id));
      }

      break;
    }
  }

  return NextResponse.json({ received: true });
}
