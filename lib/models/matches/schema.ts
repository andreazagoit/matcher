import { pgTable, uuid, timestamp, text, index, uniqueIndex } from "drizzle-orm/pg-core";
import { users } from "@/lib/models/users/schema";
import { relations } from "drizzle-orm";

export const swipeActionEnum = ["like", "skip", "superlike"] as const;

/**
 * Swipes table (Discovery Actions)
 * 
 * Logs a directional intent or rejection from a source user to a target user.
 * Used heavily by the Discovery algorithm to filter out users who have already
 * been graded ('like' or 'skip') from future recommendation lists.
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
        // Prevent duplicate actions from A to B (e.g. liking twice)
        uniqueIndex("swipes_source_target_uidx").on(table.sourceUserId, table.targetUserId)
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

export type SwipeAction = typeof swipes.$inferSelect;
export type NewSwipeAction = typeof swipes.$inferInsert;
