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
  getSpaceEvents,
  getUpcomingEventsForUser,
  getEventAttendees,
  getEventsByTags,
  respondToEvent,
  markEventCompleted,
  getEventById,
  getAccessibleSpaceIds,
  getMyAttendeeStatus,
} from "./operations";
import { events, eventAttendees } from "./schema";
import { spaces } from "@/lib/models/spaces/schema";
import { getUserById } from "@/lib/models/users/operations";
import { generateEmbedding, computeCentroid } from "@/lib/embeddings";
import { boostInterestsFromTags } from "@/lib/models/interests/operations";
import { getUserInterestTags } from "@/lib/models/interests/operations";
import { db } from "@/lib/db/drizzle";
import { embeddings } from "@/lib/models/embeddings/schema";
import { getStoredEmbedding } from "@/lib/models/embeddings/operations";
import { eq, sql, gte, and, inArray } from "drizzle-orm";

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

    spaceEvents: async (
      _: unknown,
      { spaceId }: { spaceId: string },
      context: GraphQLContext,
    ) => {
      requireAuth(context);
      const userId = context.auth.user!.id;
      return await getSpaceEvents(spaceId, userId);
    },

    myUpcomingEvents: async (
      _: unknown,
      __: unknown,
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) return [];
      return await getUpcomingEventsForUser(context.auth.user.id);
    },

    eventAttendees: async (
      _: unknown,
      { eventId }: { eventId: string },
      context: GraphQLContext,
    ) => {
      requireAuth(context);
      return await getEventAttendees(eventId);
    },

    eventsByTags: async (
      _: unknown,
      { tags, matchAll }: { tags: string[]; matchAll?: boolean },
      context: GraphQLContext,
    ) => {
      requireAuth(context);
      const userId = context.auth.user!.id;
      return await getEventsByTags(tags, matchAll ?? false, userId);
    },

    recommendedEvents: async (
      _: unknown,
      { limit }: { limit?: number },
      context: GraphQLContext,
    ) => {
      const maxResults = limit ?? 10;
      const userId = context.auth.user?.id;

      // Build accessible space filter once
      const accessibleIds = await getAccessibleSpaceIds(userId);
      const accessFilter = Array.isArray(accessibleIds) && accessibleIds.length > 0
        ? inArray(events.spaceId, accessibleIds)
        : eq(events.spaceId, "00000000-0000-0000-0000-000000000000"); // no results if nothing accessible

      const baseConditions = and(
        eq(events.status, "published"),
        gte(events.startsAt, new Date()),
        accessFilter,
      );

      if (!userId) {
        return await db.query.events.findMany({
          where: baseConditions,
          orderBy: [sql`${events.startsAt} ASC`],
          limit: maxResults,
        });
      }

      const userTags = await getUserInterestTags(userId);
      const userEmbedding = await getStoredEmbedding(userId, "user");

      // Strategy 1: OpenAI behavioral embedding
      if (userEmbedding) {
        const embeddingStr = `[${userEmbedding.join(",")}]`;
        const results = await db
          .select()
          .from(events)
          .where(and(baseConditions, sql`${events.embedding} IS NOT NULL`))
          .orderBy(sql`${events.embedding} <=> ${embeddingStr}::vector`)
          .limit(maxResults);
        if (results.length > 0) return results;
      }

      // Strategy 2: tag overlap (cold start)
      if (userTags.length > 0) {
        const tagArray = `{${userTags.join(",")}}`;
        const results = await db
          .select()
          .from(events)
          .where(and(baseConditions, sql`${events.tags} && ${tagArray}::text[]`))
          .orderBy(sql`${events.startsAt} ASC`)
          .limit(maxResults);
        if (results.length > 0) return results;
      }

      // Strategy 3: chronological fallback
      return await db.query.events.findMany({
        where: baseConditions,
        orderBy: [sql`${events.startsAt} ASC`],
        limit: maxResults,
      });
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
          tags?: string[];
        };
      },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);

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
        tags: input.tags,
        createdBy: user.id,
      });

      // Generate OpenAI embedding in background (non-blocking)
      const embText = [event.title, event.description, event.tags?.length ? `Tags: ${event.tags.join(", ")}` : ""].filter(Boolean).join("\n");
      if (embText.trim()) {
        generateEmbedding(embText)
          .then((embedding) => db.update(events).set({ embedding }).where(eq(events.id, event.id)))
          .catch(() => {});
      }

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
          tags?: string[];
          status?: "draft" | "published" | "cancelled" | "completed";
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

      if (event.createdBy !== user.id) {
        throw new GraphQLError("Only the event creator can update this event", {
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
        tags: input.tags,
        status: input.status,
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

      const result = await respondToEvent(eventId, user.id, status);

      // Boost interests from event tags in background
      if (status === "going" || status === "attended") {
        if (event.tags?.length) {
          boostInterestsFromTags(user.id, event.tags, 0.1).catch(() => {});
        }
      }

      // Recompute OpenAI behavioral centroid in background
      (async () => {
        const attended = await db
          .select({ eventId: eventAttendees.eventId })
          .from(eventAttendees)
          .where(and(eq(eventAttendees.userId, user.id), inArray(eventAttendees.status, ["going", "attended"])));
        if (attended.length === 0) return;
        const eventRows = await db.select({ embedding: events.embedding }).from(events).where(inArray(events.id, attended.map((a) => a.eventId)));
        const validEmbeddings = eventRows.map((r) => r.embedding).filter((e): e is number[] => e !== null && e.length > 0);
        if (validEmbeddings.length === 0) return;
        const centroid = computeCentroid(validEmbeddings);
        await db
          .insert(embeddings)
          .values({ entityId: user.id, entityType: "user", embedding: centroid })
          .onConflictDoUpdate({
            target: [embeddings.entityId, embeddings.entityType],
            set: { embedding: centroid, updatedAt: new Date() },
          });
      })().catch(() => {});

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

    /** The space this event belongs to */
    space: async (event: { spaceId: string }) => {
      return await db.query.spaces.findFirst({
        where: eq(spaces.id, event.spaceId),
      });
    },
  },

  EventAttendee: {
    user: async (attendee: { userId: string }) => {
      return await getUserById(attendee.userId);
    },
  },
};
