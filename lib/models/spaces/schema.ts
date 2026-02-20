import {
  pgTable,
  uuid,
  text,
  timestamp,
  boolean,
  index,
} from "drizzle-orm/pg-core";
import { vector } from "drizzle-orm/pg-core/columns/vector_extension/vector";
import { relations } from "drizzle-orm";
import { members } from "@/lib/models/members/schema";
import { users } from "@/lib/models/users/schema";

const EMBEDDING_DIMENSIONS = 1536;

/**
 * Spaces Schema
 * 
 * Represents a community/club/space.
 */

export const spaces = pgTable(
  "spaces",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    // Branding
    name: text("name").notNull(),
    slug: text("slug").unique().notNull(),
    description: text("description"),
    image: text("image"),

    // Visibility & Access
    visibility: text("visibility", { enum: ["public", "private", "hidden"] }).default("public").notNull(),
    type: text("type", { enum: ["free", "tiered"] }).default("free").notNull(),
    joinPolicy: text("join_policy", { enum: ["open", "apply", "invite_only"] }).default("open").notNull(),

    // Status
    isActive: boolean("is_active").default(true).notNull(),

    // Stripe Connect
    stripeAccountId: text("stripe_account_id"),
    stripeAccountEnabled: boolean("stripe_account_enabled").default(false).notNull(),

    // Ownership: one space has exactly one owner.
    ownerId: uuid("owner_id")
      .notNull()
      .references(() => users.id, { onDelete: "restrict" }),

    // Tags (shared vocabulary from models/tags/data.ts)
    tags: text("tags").array().default([]),

    // AI embedding for recommendations (name + description + tags)
    embedding: vector("embedding", { dimensions: EMBEDDING_DIMENSIONS }),

    // Timestamps
    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [
    index("spaces_slug_idx").on(table.slug),
    index("spaces_tags_idx").using("gin", table.tags),
  ]
);

// Relations
export const spacesRelations = relations(spaces, ({ many }) => ({
  members: many(members),
}));

// Types
export type Space = typeof spaces.$inferSelect;
export type NewSpace = typeof spaces.$inferInsert;
