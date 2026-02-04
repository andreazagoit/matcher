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
import { users } from "@/lib/models/users/schema";

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
    logoUrl: text("logo_url"),

    // OAuth credentials (inherited from Apps)
    clientId: text("client_id").unique().notNull(),
    secretKey: text("secret_key").notNull(),
    secretKeyHash: text("secret_key_hash").notNull(),
    redirectUris: text("redirect_uris").array(),

    // Settings
    isPublic: boolean("is_public").default(true).notNull(),
    requiresApproval: boolean("requires_approval").default(false).notNull(),

    // Stats
    membersCount: integer("members_count").default(0),

    // Token settings
    accessTokenTtl: text("access_token_ttl").default("3600"),
    refreshTokenTtl: text("refresh_token_ttl").default("2592000"),

    // Owner
    ownerId: uuid("owner_id").notNull().references(() => users.id, { onDelete: "cascade" }),

    // Status
    isActive: boolean("is_active").default(true).notNull(),

    // Timestamps
    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [
    index("spaces_client_id_idx").on(table.clientId),
    index("spaces_owner_idx").on(table.ownerId),
    index("spaces_slug_idx").on(table.slug),
  ]
);

// Relations
export const spacesRelations = relations(spaces, ({ one, many }) => ({
  owner: one(users, {
    fields: [spaces.ownerId],
    references: [users.id],
  }),
  // Members reference will be added in members schema
  // Posts reference will be added in posts schema
}));

// Types
export type Space = typeof spaces.$inferSelect;
export type NewSpace = typeof spaces.$inferInsert;
