import { pgTable, uuid, timestamp, real, text, index, uniqueIndex } from "drizzle-orm/pg-core";
import { users } from "@/lib/models/users/schema";
import { relations } from "drizzle-orm";

export const swipeActionEnum = ["like", "skip", "superlike"] as const;

/**
 * Swipes table (Discovery Actions)
 */
export const swipes = pgTable(
    "swipes",
    {
        id: uuid("id").primaryKey().defaultRandom(),
        sourceUserId: uuid("source_user_id")
            .notNull()
            .references(() => users.id, { onDelete: "cascade" }),
        targetUserId: uuid("target_user_id")
            .notNull()
            .references(() => users.id, { onDelete: "cascade" }),
        action: text("action", { enum: swipeActionEnum }).notNull(),
        createdAt: timestamp("created_at").defaultNow().notNull(),
    },
    (table) => [
        index("swipes_source_idx").on(table.sourceUserId),
        index("swipes_target_idx").on(table.targetUserId),
        index("swipes_action_idx").on(table.action),
        uniqueIndex("swipes_source_target_uidx").on(table.sourceUserId, table.targetUserId),
    ],
);

export const swipesRelations = relations(swipes, ({ one }) => ({
    sourceUser: one(users, {
        fields: [swipes.sourceUserId],
        references: [users.id],
        relationName: "swipes_given",
    }),
    targetUser: one(users, {
        fields: [swipes.targetUserId],
        references: [users.id],
        relationName: "swipes_received",
    }),
}));

/**
 * Daily matches — 8 pre-computed matches per user, reset every night via cron.
 * The `date` column stores the UTC date string (YYYY-MM-DD) so it's cheap to query.
 */
export const dailyMatches = pgTable(
    "daily_matches",
    {
        id: uuid("id").primaryKey().defaultRandom(),
        userId: uuid("user_id")
            .notNull()
            .references(() => users.id, { onDelete: "cascade" }),
        matchedUserId: uuid("matched_user_id")
            .notNull()
            .references(() => users.id, { onDelete: "cascade" }),
        score: real("score").notNull().default(0),
        distanceKm: real("distance_km"),
        date: text("date").notNull(), // YYYY-MM-DD UTC
        createdAt: timestamp("created_at").defaultNow().notNull(),
    },
    (table) => [
        index("daily_matches_user_date_idx").on(table.userId, table.date),
        uniqueIndex("daily_matches_user_matched_date_uidx").on(table.userId, table.matchedUserId, table.date),
    ],
);

export const dailyMatchesRelations = relations(dailyMatches, ({ one }) => ({
    user: one(users, {
        fields: [dailyMatches.userId],
        references: [users.id],
        relationName: "daily_matches_owner",
    }),
    matchedUser: one(users, {
        fields: [dailyMatches.matchedUserId],
        references: [users.id],
        relationName: "daily_matches_target",
    }),
}));

export type SwipeAction = typeof swipes.$inferSelect;
export type NewSwipeAction = typeof swipes.$inferInsert;
export type DailyMatch = typeof dailyMatches.$inferSelect;
