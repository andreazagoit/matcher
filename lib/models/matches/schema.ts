import { pgTable, uuid, timestamp, primaryKey } from "drizzle-orm/pg-core";
import { users } from "@/lib/models/users/schema";
import { relations } from "drizzle-orm";

export const dailyMatches = pgTable(
    "daily_matches",
    {
        userId: uuid("user_id")
            .notNull()
            .references(() => users.id, { onDelete: "cascade" }),

        matchId: uuid("match_id")
            .notNull()
            .references(() => users.id, { onDelete: "cascade" }),

        createdAt: timestamp("created_at").defaultNow().notNull(),
    },
    (table) => [
        primaryKey({ columns: [table.userId, table.matchId] }),
    ]
);

export const dailyMatchesRelations = relations(dailyMatches, ({ one }) => ({
    user: one(users, {
        fields: [dailyMatches.userId],
        references: [users.id],
        relationName: "daily_matches_for",
    }),
    match: one(users, {
        fields: [dailyMatches.matchId],
        references: [users.id],
        relationName: "daily_match_candidate",
    }),
}));

export type DailyMatch = typeof dailyMatches.$inferSelect;
export type NewDailyMatch = typeof dailyMatches.$inferInsert;
