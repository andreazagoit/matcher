import {
  pgTable,
  uuid,
  text,
  timestamp,
  index,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { users } from "@/lib/models/users/schema";

/**
 * Impressions — behavioral interaction log for ML training.
 *
 * Every time the app shows or the user interacts with an entity
 * (user, event, space), an impression is recorded.
 *
 * This is the primary training signal for the recommendation model.
 * Positive signals: clicked, joined, messaged.
 * Negative signals: skipped, shown (with no follow-up action).
 */
export const impressionActionEnum = [
  "shown",      // item was displayed to user (no action taken = implicit negative)
  "clicked",    // user opened the item detail
  "skipped",    // user explicitly dismissed/swiped away
  "joined",     // user joined a space or RSVP'd to an event
  "messaged",   // user sent a message request to a person
] as const;

export const impressionItemTypeEnum = ["user", "event", "space"] as const;

export const impressions = pgTable(
  "impressions",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    userId: uuid("user_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),

    itemId: uuid("item_id").notNull(),

    itemType: text("item_type", { enum: impressionItemTypeEnum }).notNull(),

    action: text("action", { enum: impressionActionEnum }).notNull(),

    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [
    index("impressions_user_idx").on(table.userId),
    index("impressions_item_idx").on(table.itemId),
    index("impressions_user_item_idx").on(table.userId, table.itemId),
    index("impressions_created_at_idx").on(table.createdAt),
  ],
);

export const impressionsRelations = relations(impressions, ({ one }) => ({
  user: one(users, {
    fields: [impressions.userId],
    references: [users.id],
  }),
}));

export type Impression = typeof impressions.$inferSelect;
export type NewImpression = typeof impressions.$inferInsert;
