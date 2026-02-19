import {
  pgTable,
  uuid,
  timestamp,
  index,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { vector } from "drizzle-orm/pg-core/columns/vector_extension/vector";
import { users } from "@/lib/models/users/schema";

const EMBEDDING_DIMENSIONS = 1536;

/**
 * User Profiles — behavioral embedding for matching.
 *
 * Each user has exactly one profile (1:1 with user).
 * The behavioral embedding is a centroid of attended event embeddings,
 * updated automatically when the user RSVPs to events.
 *
 * Interest tags are stored separately in the `user_interests` table
 * with individual weights that evolve over time.
 */
export const profiles = pgTable(
  "profiles",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    userId: uuid("user_id")
      .notNull()
      .unique()
      .references(() => users.id, { onDelete: "cascade" }),

    behaviorEmbedding: vector("behavior_embedding", {
      dimensions: EMBEDDING_DIMENSIONS,
    }),

    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [
    index("profiles_user_idx").on(table.userId),
  ],
);

export const profilesRelations = relations(profiles, ({ one }) => ({
  user: one(users, {
    fields: [profiles.userId],
    references: [users.id],
  }),
}));

export const usersRelations = relations(users, ({ one }) => ({
  profile: one(profiles, {
    fields: [users.id],
    references: [profiles.userId],
  }),
}));

// Inferred types
export type Profile = typeof profiles.$inferSelect;
export type NewProfile = typeof profiles.$inferInsert;
