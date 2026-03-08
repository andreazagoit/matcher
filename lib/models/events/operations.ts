import { db } from "@/lib/db/drizzle";
import {
  events,
  eventAttendees,
  type Event,
  type EventAttendee,
} from "./schema";
import { spaces } from "@/lib/models/spaces/schema";
import { members } from "@/lib/models/members/schema";
import { embeddings } from "@/lib/models/embeddings/schema";
import { eq, and, ne, gte, desc, asc, sql, inArray } from "drizzle-orm";
import { recordImpression } from "@/lib/models/impressions/operations";
import { embedEvent } from "@/lib/ml/client";
import type { CreateEventInput, UpdateEventInput } from "./validator";

// ─── Access Control ─────────────────────────────────────────────────

/**
 * Returns space IDs accessible to a user:
 *  - all public spaces (visibility = 'public')
 *  - private/hidden spaces where the user has an active membership
 */
export async function getAccessibleSpaceIds(userId?: string): Promise<string[] | "public_only"> {
  const publicSpaces = await db
    .select({ id: spaces.id })
    .from(spaces)
    .where(eq(spaces.visibility, "public"));

  const publicIds = publicSpaces.map((s) => s.id);

  if (!userId) return publicIds.length > 0 ? publicIds : "public_only";

  const memberSpaces = await db
    .select({ spaceId: members.spaceId })
    .from(members)
    .where(and(eq(members.userId, userId), eq(members.status, "active")));

  const memberIds = memberSpaces.map((m) => m.spaceId);

  // Merge + deduplicate
  const all = Array.from(new Set([...publicIds, ...memberIds]));
  return all;
}

/**
 * Check if a user can access events of a given space.
 * Public spaces → always yes.
 * Private/hidden → only active members.
 */
export async function canAccessSpace(spaceId: string, userId?: string): Promise<boolean> {
  const space = await db.query.spaces.findFirst({
    where: eq(spaces.id, spaceId),
    columns: { visibility: true },
  });

  if (!space) return false;
  if (space.visibility === "public") return true;
  if (!userId) return false;

  const membership = await db.query.members.findFirst({
    where: and(
      eq(members.spaceId, spaceId),
      eq(members.userId, userId),
      eq(members.status, "active"),
    ),
    columns: { id: true },
  });

  return !!membership;
}

// ─── Events CRUD ───────────────────────────────────────────────────

export async function createEvent(data: CreateEventInput & { createdBy: string }): Promise<Event> {
  const [event] = await db
    .insert(events)
    .values({
      spaceId: data.spaceId,
      title: data.title,
      description: data.description,
      location: data.location,
      coordinates: data.coordinates
        ? { x: data.coordinates.lon, y: data.coordinates.lat }
        : null,
      startsAt: data.startsAt,
      endsAt: data.endsAt,
      maxAttendees: data.maxAttendees,
      categories: data.categories ?? [],
      price: data.price ?? null,
      currency: data.currency ?? "eur",
      cover: data.cover,
      images: data.images ?? [],
      createdBy: data.createdBy,
    })
    .returning();

  // Generate and store embedding
  const embedding = await embedEvent({
    categories: data.categories,
    startsAt: data.startsAt,
    attendeeCount: 0,
    maxAttendees: data.maxAttendees,
    isPaid: (data.price ?? 0) > 0,
    priceCents: data.price ?? null,
  });
  if (embedding) {
    await db
      .insert(embeddings)
      .values({ entityId: event.id, entityType: "event", embedding })
      .onConflictDoUpdate({
        target: [embeddings.entityId, embeddings.entityType],
        set: { embedding, updatedAt: new Date() },
      });
  }

  return event;
}

export async function updateEvent(id: string, data: UpdateEventInput): Promise<Event> {
  const { coordinates, cover, images, ...rest } = data;
  const [updated] = await db
    .update(events)
    .set({
      ...rest,
      ...(coordinates !== undefined && {
        coordinates: coordinates
          ? { x: coordinates.lon, y: coordinates.lat }
          : null,
      }),
      ...(cover !== undefined && { cover }),
      ...(images !== undefined && { images }),
      updatedAt: new Date(),
    })
    .where(eq(events.id, id))
    .returning();


  if (!updated) throw new Error("Event not found");

  // Regenerate embedding if relevant fields changed
  if (data.categories !== undefined || data.startsAt !== undefined || data.price !== undefined) {
    const embedding = await embedEvent({
      categories: updated.categories ?? [],
      startsAt: updated.startsAt,
      attendeeCount: 0,
      maxAttendees: updated.maxAttendees,
      isPaid: (updated.price ?? 0) > 0,
      priceCents: updated.price ?? null,
    });
    if (embedding) {
      await db
        .insert(embeddings)
        .values({ entityId: updated.id, entityType: "event", embedding })
        .onConflictDoUpdate({
          target: [embeddings.entityId, embeddings.entityType],
          set: { embedding, updatedAt: new Date() },
        });
    }
  }

  return updated;
}

/**
 * Fetch a single event. Returns null if the event doesn't exist
 * or if the user doesn't have access to the parent space.
 */
export async function getEventById(id: string, userId?: string): Promise<Event | null> {
  const event = await db.query.events.findFirst({
    where: eq(events.id, id),
  });

  if (!event) return null;

  const accessible = await canAccessSpace(event.spaceId, userId);
  if (!accessible) return null;

  // Track the visit server-side (fire-and-forget)
  if (userId) recordImpression(userId, id, "event", "viewed");

  return event;
}

/**
 * Get events for a space (only if the user has access).
 */
export async function getSpaceEvents(spaceId: string, userId?: string, limit?: number, offset?: number): Promise<Event[]> {
  const accessible = await canAccessSpace(spaceId, userId);
  if (!accessible) return [];

  return await db.query.events.findMany({
    where: eq(events.spaceId, spaceId),
    orderBy: [asc(events.startsAt)],
    ...(limit !== undefined ? { limit } : {}),
    ...(offset !== undefined ? { offset } : {}),
  });
}

export async function getAllEvents(limit = 24, offset = 0): Promise<Event[]> {
  const now = new Date();
  return db.query.events.findMany({
    where: (e, { gte }) => gte(e.startsAt, now),
    orderBy: [asc(events.startsAt)],
    limit,
    offset,
  });
}

export async function getUpcomingEventsForUser(
  userId: string,
): Promise<Event[]> {
  const now = new Date();
  const attendeeRows = await db
    .select({ eventId: eventAttendees.eventId })
    .from(eventAttendees)
    .where(
      and(
        eq(eventAttendees.userId, userId),
        sql`${eventAttendees.status} IN ('going', 'interested')`,
      ),
    );

  if (attendeeRows.length === 0) return [];

  const eventIds = attendeeRows.map((r) => r.eventId);
  return await db.query.events.findMany({
    where: and(
      inArray(events.id, eventIds),
      gte(events.startsAt, now),
    ),
    orderBy: [asc(events.startsAt)],
  });
}

// ─── Search by categories ───────────────────────────────────────────

export async function getEventsByCategories(
  categories: string[],
  matchAll: boolean = false,
  userId?: string,
): Promise<Event[]> {
  if (categories.length === 0) return [];

  const accessibleIds = await getAccessibleSpaceIds(userId);
  if (typeof accessibleIds !== "string" && accessibleIds.length === 0) return [];

  const catArray = `{${categories.join(",")}}`;
  const operator = matchAll ? "@>" : "&&";

  const accessFilter =
    typeof accessibleIds === "string"
      ? sql`${events.spaceId} IN (SELECT id FROM spaces WHERE visibility = 'public')`
      : sql`${events.spaceId} IN (${sql.join(accessibleIds.map((id) => sql`${id}`), sql`, `)})`;

  return await db
    .select()
    .from(events)
    .where(
      and(
        sql`${events.categories} ${sql.raw(operator)} ${catArray}::text[]`,
        gte(events.startsAt, new Date()),
        accessFilter,
      ),
    )
    .orderBy(asc(events.startsAt));
}

// ─── Attendees ─────────────────────────────────────────────────────

export async function respondToEvent(
  eventId: string,
  userId: string,
  status: "going" | "interested" | "attended",
): Promise<EventAttendee> {
  const [attendee] = await db
    .insert(eventAttendees)
    .values({
      eventId,
      userId,
      status,
    })
    .onConflictDoUpdate({
      target: [eventAttendees.eventId, eventAttendees.userId],
      set: { status },
    })
    .returning();
  return attendee;
}

export async function getEventAttendees(
  eventId: string,
): Promise<EventAttendee[]> {
  return await db.query.eventAttendees.findMany({
    where: eq(eventAttendees.eventId, eventId),
    orderBy: [desc(eventAttendees.registeredAt)],
  });
}

export async function getMyAttendeeStatus(
  eventId: string,
  userId: string,
): Promise<EventAttendee | null> {
  const row = await db.query.eventAttendees.findFirst({
    where: and(
      eq(eventAttendees.eventId, eventId),
      eq(eventAttendees.userId, userId),
    ),
  });
  return row ?? null;
}

export async function markEventCompleted(eventId: string): Promise<Event> {
  const [updated] = await db
    .update(events)
    .set({
      updatedAt: new Date(),
    })
    .where(eq(events.id, eventId))
    .returning();

  if (!updated) throw new Error("Event not found");

  // Mark all "going" attendees as "attended"
  await db
    .update(eventAttendees)
    .set({
      status: "attended",
      attendedAt: new Date(),
    })
    .where(
      and(
        eq(eventAttendees.eventId, eventId),
        eq(eventAttendees.status, "going"),
      ),
    );

  return updated;
}

/**
 * AI-recommended events similar to a given event — cosine similarity between
 * this event's embedding and all other upcoming event embeddings.
 */
export async function getEventRecommendedEvents(
  eventId: string,
  limit = 6,
): Promise<Event[]> {
  const row = await db.query.embeddings.findFirst({
    where: and(
      eq(embeddings.entityId, eventId),
      eq(embeddings.entityType, "event"),
    ),
    columns: { embedding: true },
  });
  if (!row) return [];

  const vec = `[${row.embedding.join(",")}]`;
  const rows = await db
    .select({ id: events.id })
    .from(embeddings)
    .innerJoin(events, eq(sql`${events.id}::text`, embeddings.entityId))
    .where(
      and(
        eq(embeddings.entityType, "event"),
        ne(events.id, eventId),
        gte(events.startsAt, new Date()),
      ),
    )
    .orderBy(sql`${embeddings.embedding} <=> ${sql.raw(`'${vec}'::vector`)}`)
    .limit(limit);

  if (!rows.length) return [];

  const ids = rows.map((r) => r.id);
  return db.query.events.findMany({
    where: (e, { inArray }) => inArray(e.id, ids),
  });
}
