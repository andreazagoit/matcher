import {
  pgTable,
  uuid,
  text,
  timestamp,
  index,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { users } from "@/lib/models/users/schema";

/**
 * OAuth 2.0 Access Tokens
 * RFC 6749 §5.1
 */

export const accessTokens = pgTable(
  "access_tokens",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    // Token hash (never store plain tokens)
    tokenHash: text("token_hash").unique().notNull(),

    // Token identifier (jti claim, for revocation)
    jti: text("jti").unique().notNull(),

    // References
    clientId: text("client_id").notNull(),
    userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }), // NULL for client_credentials

    // Scope
    scope: text("scope").notNull(),

    // Lifecycle
    expiresAt: timestamp("expires_at").notNull(),
    revokedAt: timestamp("revoked_at"),

    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [
    index("access_tokens_hash_idx").on(table.tokenHash),
    index("access_tokens_jti_idx").on(table.jti),
    index("access_tokens_client_idx").on(table.clientId),
    index("access_tokens_user_idx").on(table.userId),
  ]
);

/**
 * OAuth 2.0 Refresh Tokens
 * RFC 6749 §6
 */

export const refreshTokens = pgTable(
  "refresh_tokens",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    // Token hash
    tokenHash: text("token_hash").unique().notNull(),

    // Token identifier
    jti: text("jti").unique().notNull(),

    // Associated access token
    accessTokenId: uuid("access_token_id").references(() => accessTokens.id, { onDelete: "cascade" }),

    // References
    clientId: text("client_id").notNull(),
    userId: uuid("user_id").references(() => users.id, { onDelete: "cascade" }),

    // Scope
    scope: text("scope").notNull(),

    // Lifecycle
    expiresAt: timestamp("expires_at").notNull(),
    revokedAt: timestamp("revoked_at"),

    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [
    index("refresh_tokens_hash_idx").on(table.tokenHash),
    index("refresh_tokens_jti_idx").on(table.jti),
    index("refresh_tokens_client_idx").on(table.clientId),
    index("refresh_tokens_user_idx").on(table.userId),
  ]
);

// Relations
export const accessTokensRelations = relations(accessTokens, ({ one }) => ({
  user: one(users, {
    fields: [accessTokens.userId],
    references: [users.id],
  }),
}));

export const refreshTokensRelations = relations(refreshTokens, ({ one }) => ({
  user: one(users, {
    fields: [refreshTokens.userId],
    references: [users.id],
  }),
  accessToken: one(accessTokens, {
    fields: [refreshTokens.accessTokenId],
    references: [accessTokens.id],
  }),
}));

// Types
export type AccessToken = typeof accessTokens.$inferSelect;
export type NewAccessToken = typeof accessTokens.$inferInsert;
export type RefreshToken = typeof refreshTokens.$inferSelect;
export type NewRefreshToken = typeof refreshTokens.$inferInsert;


