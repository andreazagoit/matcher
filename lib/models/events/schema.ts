import {
  pgTable,
  uuid,
  text,
  timestamp,
  integer,
  boolean,
  index,
  uniqueIndex,
  pgEnum,
  geometry,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { vector } from "drizzle-orm/pg-core/columns/vector_extension/vector";
import { spaces } from "@/lib/models/spaces/schema";
import { users } from "@/lib/models/users/schema";

const EMBEDDING_DIMENSIONS = 1536;

// ─── Enums ─────────────────────────────────────────────────────────

export const eventStatusEnum = pgEnum("event_status", [
  "draft",
  "published",
  "cancelled",
  "completed",
]);

export const attendeeStatusEnum = pgEnum("attendee_status", [
  "going",
  "interested",
  "attended",
]);

// ─── Events ────────────────────────────────────────────────────────

export const events = pgTable(
  "events",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    spaceId: uuid("space_id")
      .notNull()
      .references(() => spaces.id, { onDelete: "cascade" }),

    title: text("title").notNull(),
    description: text("description"),
    location: text("location"),
    coordinates: geometry("coordinates", { type: "point", mode: "xy", srid: 4326 }),

    startsAt: timestamp("starts_at").notNull(),
    endsAt: timestamp("ends_at"),

    maxAttendees: integer("max_attendees"),
    status: eventStatusEnum("status").default("draft").notNull(),

    // Ticketing — null means free event
    price: integer("price"),
    currency: text("currency").default("eur"),

    // Tags (shared vocabulary from models/tags/data.ts)
    tags: text("tags").array().default([]),

    embedding: vector("embedding", { dimensions: EMBEDDING_DIMENSIONS }),

    createdBy: uuid("created_by")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),

    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [
    index("events_space_idx").on(table.spaceId),
    index("events_starts_at_idx").on(table.startsAt),
    index("events_created_by_idx").on(table.createdBy),
    index("events_coordinates_gist_idx").using("gist", table.coordinates),
    index("events_tags_idx").using("gin", table.tags),
  ],
);

// ─── Event Attendees ───────────────────────────────────────────────

export const eventAttendees = pgTable(
  "event_attendees",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    eventId: uuid("event_id")
      .notNull()
      .references(() => events.id, { onDelete: "cascade" }),

    userId: uuid("user_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),

    status: attendeeStatusEnum("status").default("going").notNull(),
    registeredAt: timestamp("registered_at").defaultNow().notNull(),
    attendedAt: timestamp("attended_at"),

    // Stripe payment tracking
    paymentStatus: text("payment_status", { enum: ["pending", "paid", "refunded"] }),
    stripeCheckoutSessionId: text("stripe_checkout_session_id"),
  },
  (table) => [
    uniqueIndex("event_attendees_event_user_idx").on(
      table.eventId,
      table.userId,
    ),
    index("event_attendees_user_idx").on(table.userId),
  ],
);

// ─── Relations ─────────────────────────────────────────────────────

export const eventsRelations = relations(events, ({ one, many }) => ({
  space: one(spaces, {
    fields: [events.spaceId],
    references: [spaces.id],
  }),
  creator: one(users, {
    fields: [events.createdBy],
    references: [users.id],
  }),
  attendees: many(eventAttendees),
}));

export const eventAttendeesRelations = relations(eventAttendees, ({ one }) => ({
  event: one(events, {
    fields: [eventAttendees.eventId],
    references: [events.id],
  }),
  user: one(users, {
    fields: [eventAttendees.userId],
    references: [users.id],
  }),
}));

// ─── Types ─────────────────────────────────────────────────────────

export type Event = typeof events.$inferSelect;
export type NewEvent = typeof events.$inferInsert;
export type EventAttendee = typeof eventAttendees.$inferSelect;
export type NewEventAttendee = typeof eventAttendees.$inferInsert;
