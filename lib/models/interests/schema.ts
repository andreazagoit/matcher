import {
  pgTable,
  uuid,
  real,
  text,
  timestamp,
  index,
  uniqueIndex,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { users } from "@/lib/models/users/schema";

/**
 * User Interests — weighted, evolving interest tags.
 *
 * Each user can have many interests, each with a weight (0.0–1.0)
 * that evolves based on behavior: events attended, spaces joined,
 * conversations accepted.
 */
export const userInterests = pgTable(
  "user_interests",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    userId: uuid("user_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),

    tag: text("tag").notNull(),

    weight: real("weight").default(1.0).notNull(),

    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [
    uniqueIndex("user_interests_user_tag_idx").on(table.userId, table.tag),
    index("user_interests_user_idx").on(table.userId),
    index("user_interests_tag_idx").on(table.tag),
  ],
);

export const userInterestsRelations = relations(userInterests, ({ one }) => ({
  user: one(users, {
    fields: [userInterests.userId],
    references: [users.id],
  }),
}));

export type UserInterest = typeof userInterests.$inferSelect;
export type NewUserInterest = typeof userInterests.$inferInsert;
