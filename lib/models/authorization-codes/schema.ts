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
 * OAuth 2.0 Authorization Codes
 * RFC 6749 §4.1 + RFC 7636 (PKCE)
 * 
 * Short-lived codes (~10 min) exchanged for tokens
 */

export const authorizationCodes = pgTable(
  "authorization_codes",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    // The authorization code
    code: text("code").unique().notNull(),

    // References
    clientId: text("client_id").notNull(),
    userId: uuid("user_id").notNull().references(() => users.id, { onDelete: "cascade" }),

    // Request details
    redirectUri: text("redirect_uri").notNull(),
    scope: text("scope").notNull(),
    state: text("state"),

    // PKCE (RFC 7636)
    codeChallenge: text("code_challenge"),
    codeChallengeMethod: text("code_challenge_method"), // 'S256' | 'plain'

    // Lifecycle
    expiresAt: timestamp("expires_at").notNull(),
    usedAt: timestamp("used_at"), // NULL = not used yet

    createdAt: timestamp("created_at").defaultNow().notNull(),
  },
  (table) => [
    index("authorization_codes_code_idx").on(table.code),
    index("authorization_codes_client_idx").on(table.clientId),
    index("authorization_codes_user_idx").on(table.userId),
  ]
);

// Relations
export const authorizationCodesRelations = relations(authorizationCodes, ({ one }) => ({
  user: one(users, {
    fields: [authorizationCodes.userId],
    references: [users.id],
  }),
}));

// Types
export type AuthorizationCode = typeof authorizationCodes.$inferSelect;
export type NewAuthorizationCode = typeof authorizationCodes.$inferInsert;


