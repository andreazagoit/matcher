import { db } from "@/lib/db/drizzle";
import {
  events,
  eventAttendees,
  type Event,
  type EventAttendee,
} from "./schema";
import { eq, and, gte, desc, asc, sql } from "drizzle-orm";

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

export async function getEventById(id: string): Promise<Event | null> {
  const result = await db.query.events.findFirst({
    where: eq(events.id, id),
  });
  return result ?? null;
}

export async function getSpaceEvents(spaceId: string): Promise<Event[]> {
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
      sql`${events.id} IN (${sql.join(eventIds.map((id) => sql`${id}`), sql`, `)})`,
      gte(events.startsAt, now),
    ),
    orderBy: [asc(events.startsAt)],
  });
}

// ─── Search by tags ─────────────────────────────────────────────────

export async function getEventsByTags(
  tags: string[],
  matchAll: boolean = false,
): Promise<Event[]> {
  if (tags.length === 0) return [];

  const tagArray = `{${tags.join(",")}}`;
  const operator = matchAll ? "@>" : "&&";

  return await db
    .select()
    .from(events)
    .where(
      and(
        sql`${events.tags} ${sql.raw(operator)} ${tagArray}::text[]`,
        eq(events.status, "published"),
        gte(events.startsAt, new Date()),
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
