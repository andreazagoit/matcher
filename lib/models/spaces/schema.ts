import {
  pgTable,
  uuid,
  text,
  timestamp,
  boolean,
  index,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { members } from "@/lib/models/members/schema";

/**
 * Spaces Schema (ex Apps)
 * 
 * Represents a community/club/space.
 * Inherits OAuth capabilities from Apps.
 */

export const spaces = pgTable(
  "spaces",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    // Branding
    name: text("name").notNull(),
    slug: text("slug").unique().notNull(), // URL friendly identifier
    description: text("description"),
    image: text("image"),

    // OAuth credentials (inherited from Apps)
    clientId: text("client_id").unique().notNull(),
    secretKey: text("secret_key").notNull(),
    secretKeyHash: text("secret_key_hash").notNull(),
    redirectUris: text("redirect_uris").array(),

    // Visibility & Access
    visibility: text("visibility", { enum: ["public", "private", "hidden"] }).default("public").notNull(),
    type: text("type", { enum: ["free", "tiered"] }).default("free").notNull(),
    joinPolicy: text("join_policy", { enum: ["open", "apply", "invite_only"] }).default("open").notNull(),

    // Token settings
    accessTokenTtl: text("access_token_ttl").default("3600"),
    refreshTokenTtl: text("refresh_token_ttl").default("2592000"),

    // Status
    isActive: boolean("is_active").default(true).notNull(),

    // Timestamps
    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [
    index("spaces_client_id_idx").on(table.clientId),
    index("spaces_slug_idx").on(table.slug),
  ]
);

// Relations
export const spacesRelations = relations(spaces, ({ many }) => ({
  members: many(members),
}));

// Types
export type Space = typeof spaces.$inferSelect;
export type NewSpace = typeof spaces.$inferInsert;
