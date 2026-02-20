import { db } from "@/lib/db/drizzle";
import {
  events,
  eventAttendees,
  type Event,
  type EventAttendee,
} from "./schema";
import { spaces } from "@/lib/models/spaces/schema";
import { members } from "@/lib/models/members/schema";
import { eq, and, gte, desc, asc, sql, inArray } from "drizzle-orm";

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

export async function createEvent(data: {
  spaceId: string;
  title: string;
  description?: string;
  location?: string;
  coordinates?: { lat: number; lon: number };
  startsAt: Date;
  endsAt?: Date;
  maxAttendees?: number;
  status?: "draft" | "published" | "cancelled" | "completed";
  tags?: string[];
  price?: number;
  currency?: string;
  createdBy: string;
}): Promise<Event> {
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
      status: data.status || "published",
      tags: data.tags ?? [],
      price: data.price ?? null,
      currency: data.currency ?? "eur",
      createdBy: data.createdBy,
    })
    .returning();
  return event;
}

export async function updateEvent(
  id: string,
  data: {
    title?: string;
    description?: string;
    location?: string;
    coordinates?: { lat: number; lon: number } | null;
    startsAt?: Date;
    endsAt?: Date;
    maxAttendees?: number;
    status?: "draft" | "published" | "cancelled" | "completed";
    tags?: string[];
    price?: number;
    currency?: string;
  },
): Promise<Event> {
  const { coordinates, ...rest } = data;
  const [updated] = await db
    .update(events)
    .set({
      ...rest,
      ...(coordinates !== undefined && {
        coordinates: coordinates
          ? { x: coordinates.lon, y: coordinates.lat }
          : null,
      }),
      updatedAt: new Date(),
    })
    .where(eq(events.id, id))
    .returning();


  if (!updated) throw new Error("Event not found");
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

  return event;
}

/**
 * Get events for a space (only if the user has access).
 */
export async function getSpaceEvents(spaceId: string, userId?: string): Promise<Event[]> {
  const accessible = await canAccessSpace(spaceId, userId);
  if (!accessible) return [];

  return await db.query.events.findMany({
    where: eq(events.spaceId, spaceId),
    orderBy: [asc(events.startsAt)],
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

// ─── Search by tags ─────────────────────────────────────────────────

export async function getEventsByTags(
  tags: string[],
  matchAll: boolean = false,
  userId?: string,
): Promise<Event[]> {
  if (tags.length === 0) return [];

  const accessibleIds = await getAccessibleSpaceIds(userId);
  if (typeof accessibleIds !== "string" && accessibleIds.length === 0) return [];

  const tagArray = `{${tags.join(",")}}`;
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
        sql`${events.tags} ${sql.raw(operator)} ${tagArray}::text[]`,
        eq(events.status, "published"),
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
      status: "completed",
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
