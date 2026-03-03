/**
 * Event resolvers — CRUD and attendance management.
 * Visibility rules:
 *  - public space → events visible to all authenticated users
 *  - private/hidden space → events visible only to active members
 */

import { GraphQLError } from "graphql";
import type { GraphQLContext } from "@/lib/graphql/context";
import {
  createEvent,
  updateEvent,
  getUpcomingEventsForUser,
  respondToEvent,
  markEventCompleted,
  getEventById,
  getMyAttendeeStatus,
  getEventRecommendedEvents,
  getEventAttendees,
} from "./operations";
import { events } from "./schema";
import { spaces } from "@/lib/models/spaces/schema";
import { getUserById } from "@/lib/models/users/operations";
import { db } from "@/lib/db/drizzle";
import { embedEvent } from "@/lib/models/embeddings/operations";
import { eq } from "drizzle-orm";

function requireAuth(context: GraphQLContext) {
  if (!context.auth.user) {
    throw new GraphQLError("Authentication required", {
      extensions: { code: "UNAUTHENTICATED" },
    });
  }
  return context.auth.user;
}

export const eventResolvers = {
  Query: {
    /**
     * Get a single event — respects space visibility.
     * Returns null (not an error) if the event doesn't exist or is inaccessible.
     */
    event: async (
      _: unknown,
      { id }: { id: string },
      context: GraphQLContext,
    ) => {
      const userId = context.auth.user?.id;
      return await getEventById(id, userId);
    },

    myUpcomingEvents: async (
      _: unknown,
      __: unknown,
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) return [];
      return await getUpcomingEventsForUser(context.auth.user.id);
    },
  },

  Mutation: {
    createEvent: async (
      _: unknown,
      {
        input,
      }: {
        input: {
          spaceId: string;
          title: string;
          description?: string;
          location?: string;
          lat?: number;
          lon?: number;
          startsAt: string;
          endsAt?: string;
          maxAttendees?: number;
          categories?: string[];
          price?: number;
          currency?: string;
        };
      },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      const space = await db.query.spaces.findFirst({
        where: eq(spaces.id, input.spaceId),
        columns: { ownerId: true },
      });
      if (!space) {
        throw new GraphQLError("Space not found", {
          extensions: { code: "NOT_FOUND" },
        });
      }
      if (space.ownerId !== user.id) {
        throw new GraphQLError("Only the space owner can create events", {
          extensions: { code: "FORBIDDEN" },
        });
      }

      const event = await createEvent({
        spaceId: input.spaceId,
        title: input.title,
        description: input.description,
        location: input.location,
        coordinates:
          input.lat != null && input.lon != null
            ? { lat: input.lat, lon: input.lon }
            : undefined,
        startsAt: new Date(input.startsAt),
        endsAt: input.endsAt ? new Date(input.endsAt) : undefined,
        maxAttendees: input.maxAttendees,
        categories: input.categories,
        price: input.price,
        currency: input.currency,
        createdBy: user.id,
      });

      // Generate ML embedding in background (non-blocking)
      embedEvent(event.id, {
        categories: event.categories ?? [],
        startsAt: event.startsAt?.toISOString() ?? null,
        isPaid: (event.price ?? 0) > 0,
        priceCents: event.price ?? null,
      }).catch(() => { });

      return event;
    },

    updateEvent: async (
      _: unknown,
      {
        id,
        input,
      }: {
        id: string;
        input: {
          title?: string;
          description?: string;
          location?: string;
          lat?: number;
          lon?: number;
          startsAt?: string;
          endsAt?: string;
          maxAttendees?: number;
          categories?: string[];
          price?: number;
          currency?: string;
        };
      },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      // Bypass visibility check for the creator (they always have access to their own event)
      const event = await db.query.events.findFirst({ where: eq(events.id, id) });

      if (!event) {
        throw new GraphQLError("Event not found", {
          extensions: { code: "NOT_FOUND" },
        });
      }
      const space = await db.query.spaces.findFirst({
        where: eq(spaces.id, event.spaceId),
        columns: { ownerId: true },
      });
      if (!space || space.ownerId !== user.id) {
        throw new GraphQLError("Only the space owner can update this event", {
          extensions: { code: "FORBIDDEN" },
        });
      }

      return await updateEvent(id, {
        title: input.title,
        description: input.description,
        location: input.location,
        coordinates:
          input.lat != null && input.lon != null
            ? { lat: input.lat, lon: input.lon }
            : undefined,
        startsAt: input.startsAt ? new Date(input.startsAt) : undefined,
        endsAt: input.endsAt ? new Date(input.endsAt) : undefined,
        maxAttendees: input.maxAttendees,
        categories: input.categories,
        price: input.price,
        currency: input.currency,
      });
    },

    respondToEvent: async (
      _: unknown,
      { eventId, status }: { eventId: string; status: "going" | "interested" | "attended" },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);

      // Check access before allowing RSVP
      const event = await getEventById(eventId, user.id);
      if (!event) {
        throw new GraphQLError("Event not found or access denied", {
          extensions: { code: "NOT_FOUND" },
        });
      }

      // Paid events must go through the checkout flow, not direct RSVP
      if (event.price && event.price > 0 && status === "going") {
        const existingAttendee = await getMyAttendeeStatus(eventId, user.id);
        if (existingAttendee?.paymentStatus !== "paid") {
          throw new GraphQLError("Ticket purchase required", {
            extensions: { code: "PAYMENT_REQUIRED" },
          });
        }
      }

      const result = await respondToEvent(eventId, user.id, status);

      return result;
    },

    markEventCompleted: async (
      _: unknown,
      { eventId }: { eventId: string },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      const event = await db.query.events.findFirst({ where: eq(events.id, eventId) });

      if (!event) {
        throw new GraphQLError("Event not found", {
          extensions: { code: "NOT_FOUND" },
        });
      }

      if (event.createdBy !== user.id) {
        throw new GraphQLError(
          "Only the event creator can mark this event as completed",
          { extensions: { code: "FORBIDDEN" } },
        );
      }

      return await markEventCompleted(eventId);
    },
  },

  Event: {
    coordinates: (parent: { coordinates?: { x: number; y: number } | null }) => {
      if (!parent.coordinates) return null;
      return { lat: parent.coordinates.y, lon: parent.coordinates.x };
    },

    createdBy: async (
      event: { createdBy: string },
      _: unknown,
      context: GraphQLContext,
    ) => {
      return context.loaders.userLoader.load(event.createdBy);
    },

    attendees: async (event: { id: string }) => {
      return await getEventAttendees(event.id);
    },

    attendeeCount: async (event: { id: string }) => {
      const attendees = await getEventAttendees(event.id);
      return attendees.filter(
        (a) => a.status === "going" || a.status === "attended",
      ).length;
    },

    /** Status of the currently authenticated user for this event */
    myAttendeeStatus: async (
      event: { id: string },
      _: unknown,
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) return null;
      const row = await getMyAttendeeStatus(event.id, context.auth.user.id);
      return row?.status ?? null;
    },

    isPaid: (event: { price?: number | null }) => {
      return typeof event.price === "number" && event.price > 0;
    },

    myPaymentStatus: async (
      event: { id: string; price?: number | null },
      _: unknown,
      context: GraphQLContext,
    ) => {
      if (!context.auth.user || !event.price) return null;
      const row = await getMyAttendeeStatus(event.id, context.auth.user.id);
      return row?.paymentStatus ?? null;
    },

    /** The space this event belongs to */
    space: async (event: { spaceId: string }) => {
      return await db.query.spaces.findFirst({
        where: eq(spaces.id, event.spaceId),
      });
    },

    recommendedEvents: async (
      event: { id: string },
      { limit = 6 }: { limit?: number },
    ) => {
      return getEventRecommendedEvents(event.id, limit);
    },
  },

  EventAttendee: {
    user: async (attendee: { userId: string }) => {
      return await getUserById(attendee.userId);
    },
  },
};
