import {
  pgTable,
  uuid,
  text,
  timestamp,
  boolean,
  integer,
  index,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { members } from "@/lib/models/members/schema";
import { users } from "@/lib/models/users/schema";

/**
 * Spaces Schema
 * 
 * Represents a community/club/space.
 */

export const spaces = pgTable(
  "spaces",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    // Identity
    name: text("name").notNull(),
    slug: text("slug").unique().notNull(),
    description: text("description"),

    // Media
    cover: text("cover").notNull(),
    images: text("images").array().default([]),

    // Classification
    categories: text("categories").array().default([]),

    // Access
    visibility: text("visibility", { enum: ["public", "private", "hidden"] }).default("public").notNull(),
    type: text("type", { enum: ["free", "tiered"] }).default("free").notNull(),
    joinPolicy: text("join_policy", { enum: ["open", "apply", "invite_only"] }).default("open").notNull(),

    // Ownership
    ownerId: uuid("owner_id")
      .notNull()
      .references(() => users.id, { onDelete: "restrict" }),

    // Stats (denormalized for sorting/performance)
    membersCount: integer("members_count").default(0).notNull(),

    // Stripe Connect
    stripeAccountId: text("stripe_account_id"),
    stripeAccountEnabled: boolean("stripe_account_enabled").default(false).notNull(),

    // Timestamps
    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [
    index("spaces_slug_idx").on(table.slug),
    index("spaces_categories_idx").using("gin", table.categories),
  ]
);

// Relations
export const spacesRelations = relations(spaces, ({ many }) => ({
  members: many(members),
}));

// Types
export type Space = typeof spaces.$inferSelect;
export type NewSpace = typeof spaces.$inferInsert;
