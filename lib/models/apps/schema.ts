import {
  pgTable,
  uuid,
  text,
  timestamp,
  boolean,
  index,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { users } from "@/lib/models/users/schema";

/**
 * OAuth Apps Schema
 * 
 * Hybrid approach:
 * - client_id: For OAuth flows (authorization_code)
 * - secret_key: For direct M2M API access (no OAuth flow needed)
 */

export const apps = pgTable(
  "apps",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    // App identification
    name: text("name").notNull(),
    description: text("description"),

    // OAuth credentials (for authorization_code flow)
    clientId: text("client_id").unique().notNull(),

    // M2M API Key (stored in plain, shown in dashboard)
    secretKey: text("secret_key").notNull(),

    // M2M API Key hash (for validation)
    secretKeyHash: text("secret_key_hash").notNull(),

    // Redirect URIs for OAuth flow
    redirectUris: text("redirect_uris").array(),

    // Token settings
    accessTokenTtl: text("access_token_ttl").default("3600"), // 1 hour
    refreshTokenTtl: text("refresh_token_ttl").default("2592000"), // 30 days

    // Owner (for dashboard management)
    ownerId: uuid("owner_id").references(() => users.id, { onDelete: "set null" }),

    // Status
    isActive: boolean("is_active").default(true).notNull(),

    // Timestamps
    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [
    index("apps_client_id_idx").on(table.clientId),
    index("apps_owner_idx").on(table.ownerId),
  ]
);

// Relations
export const appsRelations = relations(apps, ({ one }) => ({
  owner: one(users, {
    fields: [apps.ownerId],
    references: [users.id],
  }),
}));

// Types
// Types
export type App = typeof apps.$inferSelect;
export type NewApp = typeof apps.$inferInsert;
