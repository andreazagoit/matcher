import { NextRequest, NextResponse } from "next/server";
import { getAuthContext } from "@/lib/auth/utils";
import { stripe, PLATFORM_FEE_PERCENT } from "@/lib/stripe";
import { db } from "@/lib/db/drizzle";
import { events, eventAttendees } from "@/lib/models/events/schema";
import { spaces } from "@/lib/models/spaces/schema";
import { eq, and } from "drizzle-orm";

export async function POST(req: NextRequest) {
  const authContext = await getAuthContext();
  if (!authContext.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { eventId } = await req.json();
  if (!eventId) {
    return NextResponse.json({ error: "eventId is required" }, { status: 400 });
  }

  const event = await db.query.events.findFirst({
    where: eq(events.id, eventId),
  });

  if (!event) {
    return NextResponse.json({ error: "Event not found" }, { status: 404 });
  }

  if (!event.price || event.price <= 0) {
    return NextResponse.json({ error: "Event is free" }, { status: 400 });
  }

  const space = await db.query.spaces.findFirst({
    where: eq(spaces.id, event.spaceId),
  });

  if (!space?.stripeAccountId || !space.stripeAccountEnabled) {
    return NextResponse.json(
      { error: "Organiser has not set up payments yet" },
      { status: 400 },
    );
  }

  // Check if the user already has a paid ticket
  const existing = await db.query.eventAttendees.findFirst({
    where: and(
      eq(eventAttendees.eventId, eventId),
      eq(eventAttendees.userId, authContext.user.id),
    ),
  });

  if (existing?.paymentStatus === "paid") {
    return NextResponse.json({ error: "Already purchased" }, { status: 400 });
  }

  const appUrl = process.env.NEXT_PUBLIC_APP_URL ?? "http://localhost:3000";
  const applicationFeeAmount = Math.round(event.price * PLATFORM_FEE_PERCENT);

  const session = await stripe.checkout.sessions.create(
    {
      mode: "payment",
      payment_method_types: ["card"],
      line_items: [
        {
          quantity: 1,
          price_data: {
            currency: event.currency ?? "eur",
            unit_amount: event.price,
            product_data: {
              name: event.title,
              description: event.description ?? undefined,
            },
          },
        },
      ],
      application_fee_amount: applicationFeeAmount,
      success_url: `${appUrl}/events/${eventId}?success=1`,
      cancel_url: `${appUrl}/events/${eventId}?cancelled=1`,
      metadata: {
        eventId,
        userId: authContext.user.id,
      },
    },
    { stripeAccount: space.stripeAccountId },
  );

  // Create or update the attendee record with pending payment status
  await db
    .insert(eventAttendees)
    .values({
      eventId,
      userId: authContext.user.id,
      status: "going",
      paymentStatus: "pending",
      stripeCheckoutSessionId: session.id,
    })
    .onConflictDoUpdate({
      target: [eventAttendees.eventId, eventAttendees.userId],
      set: {
        paymentStatus: "pending",
        stripeCheckoutSessionId: session.id,
      },
    });

  return NextResponse.json({ url: session.url });
}
