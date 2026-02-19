/**
 * Event resolvers — CRUD and attendance management.
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
} from "./operations";
import { events, eventAttendees } from "./schema";
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
    spaceEvents: async (
      _: unknown,
      { spaceId }: { spaceId: string },
      context: GraphQLContext,
    ) => {
      requireAuth(context);
      return await getSpaceEvents(spaceId);
    },

    myUpcomingEvents: async (
      _: unknown,
      __: unknown,
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      return await getUpcomingEventsForUser(user.id);
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
      return await getEventsByTags(tags, matchAll ?? false);
    },

    recommendedEvents: async (
      _: unknown,
      { limit }: { limit?: number },
      context: GraphQLContext,
    ) => {
      const user = requireAuth(context);
      const maxResults = limit ?? 10;

      const userTags = await getUserInterestTags(user.id);
      const userEmbedding = await getStoredEmbedding(user.id, "user");

      // Strategy 1: OpenAI behavioral embedding
      if (userEmbedding) {
        const embeddingStr = `[${userEmbedding.join(",")}]`;
        const results = await db
          .select()
          .from(events)
          .where(
            and(
              eq(events.status, "published"),
              gte(events.startsAt, new Date()),
              sql`${events.embedding} IS NOT NULL`,
            ),
          )
          .orderBy(sql`${events.embedding} <=> ${embeddingStr}::vector`)
          .limit(maxResults);
        return results;
      }

      // Strategy 2: tag overlap (cold start)
      if (userTags.length > 0) {
        const tagArray = `{${userTags.join(",")}}`;
        const results = await db
          .select()
          .from(events)
          .where(
            and(
              eq(events.status, "published"),
              gte(events.startsAt, new Date()),
              sql`${events.tags} && ${tagArray}::text[]`,
            ),
          )
          .orderBy(sql`${events.startsAt} ASC`)
          .limit(maxResults);
        if (results.length > 0) return results;
      }

      // Strategy 3: chronological fallback
      return await db.query.events.findMany({
        where: and(
          eq(events.status, "published"),
          gte(events.startsAt, new Date()),
        ),
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
      const event = await getEventById(id);

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
      const result = await respondToEvent(eventId, user.id, status);

      // Boost interests from event tags in background
      if (status === "going" || status === "attended") {
        getEventById(eventId)
          .then((ev) => {
            if (ev?.tags?.length) {
              boostInterestsFromTags(user.id, ev.tags, 0.1);
            }
          })
          .catch(() => {});
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
      const event = await getEventById(eventId);

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
  },

  EventAttendee: {
    user: async (attendee: { userId: string }) => {
      return await getUserById(attendee.userId);
    },
  },
};
