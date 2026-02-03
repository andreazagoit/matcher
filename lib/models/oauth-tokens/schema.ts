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

export const oauthAccessTokens = pgTable(
  "oauth_access_tokens",
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
    index("oauth_access_tokens_hash_idx").on(table.tokenHash),
    index("oauth_access_tokens_jti_idx").on(table.jti),
    index("oauth_access_tokens_client_idx").on(table.clientId),
    index("oauth_access_tokens_user_idx").on(table.userId),
  ]
);

/**
 * OAuth 2.0 Refresh Tokens
 * RFC 6749 §6
 */

export const oauthRefreshTokens = pgTable(
  "oauth_refresh_tokens",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    // Token hash
    tokenHash: text("token_hash").unique().notNull(),
    
    // Token identifier
    jti: text("jti").unique().notNull(),
    
    // Associated access token
    accessTokenId: uuid("access_token_id").references(() => oauthAccessTokens.id, { onDelete: "cascade" }),
    
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
    index("oauth_refresh_tokens_hash_idx").on(table.tokenHash),
    index("oauth_refresh_tokens_jti_idx").on(table.jti),
    index("oauth_refresh_tokens_client_idx").on(table.clientId),
    index("oauth_refresh_tokens_user_idx").on(table.userId),
  ]
);

// Relations
export const oauthAccessTokensRelations = relations(oauthAccessTokens, ({ one }) => ({
  user: one(users, {
    fields: [oauthAccessTokens.userId],
    references: [users.id],
  }),
}));

export const oauthRefreshTokensRelations = relations(oauthRefreshTokens, ({ one }) => ({
  user: one(users, {
    fields: [oauthRefreshTokens.userId],
    references: [users.id],
  }),
  accessToken: one(oauthAccessTokens, {
    fields: [oauthRefreshTokens.accessTokenId],
    references: [oauthAccessTokens.id],
  }),
}));

// Types
export type OAuthAccessToken = typeof oauthAccessTokens.$inferSelect;
export type NewOAuthAccessToken = typeof oauthAccessTokens.$inferInsert;
export type OAuthRefreshToken = typeof oauthRefreshTokens.$inferSelect;
export type NewOAuthRefreshToken = typeof oauthRefreshTokens.$inferInsert;

