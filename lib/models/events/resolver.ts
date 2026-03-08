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
  getAllEvents,
} from "./operations";
import type { CreateEventInput, UpdateEventInput } from "./validator";
import { events } from "./schema";
import { spaces } from "@/lib/models/spaces/schema";
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

    events: async (
      _: unknown,
      { limit = 24, offset = 0 }: { limit?: number; offset?: number },
    ) => {
      return getAllEvents(limit, offset);
    },
  },

  Mutation: {
    createEvent: async (
      _: unknown,
      { input }: { input: Omit<CreateEventInput, "createdBy" | "coordinates"> & { lat?: number; lon?: number } },
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
        startsAt: input.startsAt,
        endsAt: input.endsAt,
        maxAttendees: input.maxAttendees,
        categories: input.categories,
        price: input.price,
        currency: input.currency,
        cover: input.cover,
        images: input.images,
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
      { id, input }: { id: string; input: Omit<UpdateEventInput, "coordinates"> & { lat?: number; lon?: number } },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
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
        startsAt: input.startsAt,
        endsAt: input.endsAt,
        maxAttendees: input.maxAttendees,
        categories: input.categories,
        price: input.price,
        currency: input.currency,
        cover: input.cover,
        images: input.images,
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

    attendees: async (event: { id: string }, _: unknown, context: GraphQLContext) => {
      return context.loaders.eventAttendeesLoader.load(event.id);
    },

    attendeeCount: async (event: { id: string }, _: unknown, context: GraphQLContext) => {
      const attendees = await context.loaders.eventAttendeesLoader.load(event.id);
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
      const row = await context.loaders.myAttendeeStatusLoader.load(event.id);
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
      const row = await context.loaders.myAttendeeStatusLoader.load(event.id);
      return row?.paymentStatus ?? null;
    },

    /** The space this event belongs to */
    space: async (event: { spaceId: string }, _: unknown, context: GraphQLContext) => {
      return context.loaders.spaceLoader.load(event.spaceId);
    },

    recommendedEvents: async (
      event: { id: string },
      { limit = 6 }: { limit?: number },
    ) => {
      return getEventRecommendedEvents(event.id, limit);
    },
  },

    EventAttendee: {
        user: async (attendee: { userId: string }, _: unknown, context: GraphQLContext) => {
            return context.loaders.userLoader.load(attendee.userId);
        },
    },
};
