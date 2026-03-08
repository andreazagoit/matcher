import DataLoader from "dataloader";
import { db } from "@/lib/db/drizzle";
import { users, type User } from "@/lib/models/users/schema";
import { members, type Member } from "@/lib/models/members/schema";
import { spaces, type Space } from "@/lib/models/spaces/schema";
import { events, type Event } from "@/lib/models/events/schema";
import { eventAttendees, type EventAttendee } from "@/lib/models/events/schema";
import { eq, inArray, and } from "drizzle-orm";

/**
 * Creates a new set of data loaders for a single request.
 * Loaders must be per-request to avoid cross-user data leakage and stale cache.
 */
export function createDataLoaders(currentUserId?: string | null) {
    return {
        // ── Users ────────────────────────────────────────────────────────────

        userLoader: new DataLoader<string, User | null>(async (userIds) => {
            const results = await db.query.users.findMany({
                where: inArray(users.id, Array.from(userIds)),
            });
            const map = new Map(results.map((u) => [u.id, u]));
            return userIds.map((id) => map.get(id) ?? null);
        }),

        // ── Spaces ───────────────────────────────────────────────────────────

        spaceLoader: new DataLoader<string, Space | null>(async (spaceIds) => {
            const results = await db.query.spaces.findMany({
                where: inArray(spaces.id, Array.from(spaceIds)),
            });
            const map = new Map(results.map((s) => [s.id, s]));
            return spaceIds.map((id) => map.get(id) ?? null);
        }),

        myMembershipLoader: new DataLoader<string, Member | null>(async (spaceIds) => {
            if (!currentUserId) return spaceIds.map(() => null);
            const results = await db.query.members.findMany({
                where: and(
                    eq(members.userId, currentUserId),
                    inArray(members.spaceId, Array.from(spaceIds))
                ),
            });
            const map = new Map(results.map((m) => [m.spaceId, m]));
            return spaceIds.map((id) => map.get(id) ?? null);
        }),

        // ── Events ───────────────────────────────────────────────────────────

        eventLoader: new DataLoader<string, Event | null>(async (eventIds) => {
            const results = await db.query.events.findMany({
                where: inArray(events.id, Array.from(eventIds)),
            });
            const map = new Map(results.map((e) => [e.id, e]));
            return eventIds.map((id) => map.get(id) ?? null);
        }),

        /**
         * Loads all attendees for multiple events in a single batch.
         * Returns EventAttendee[] per eventId.
         */
        eventAttendeesLoader: new DataLoader<string, EventAttendee[]>(async (eventIds) => {
            const results = await db.query.eventAttendees.findMany({
                where: inArray(eventAttendees.eventId, Array.from(eventIds)),
            });
            const map = new Map<string, EventAttendee[]>();
            for (const attendee of results) {
                const list = map.get(attendee.eventId) ?? [];
                list.push(attendee);
                map.set(attendee.eventId, list);
            }
            return eventIds.map((id) => map.get(id) ?? []);
        }),

        /**
         * Loads the current user's attendee record for multiple events in a single batch.
         */
        myAttendeeStatusLoader: new DataLoader<string, EventAttendee | null>(async (eventIds) => {
            if (!currentUserId) return eventIds.map(() => null);
            const results = await db.query.eventAttendees.findMany({
                where: and(
                    eq(eventAttendees.userId, currentUserId),
                    inArray(eventAttendees.eventId, Array.from(eventIds)),
                ),
            });
            const map = new Map(results.map((a) => [a.eventId, a]));
            return eventIds.map((id) => map.get(id) ?? null);
        }),
    };
}

export type DataLoaders = ReturnType<typeof createDataLoaders>;
