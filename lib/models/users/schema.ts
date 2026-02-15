import {
  pgTable,
  uuid,
  text,
  date,
  timestamp,
  index,
  pgEnum,
  boolean,
} from "drizzle-orm/pg-core";

/**
 * Normalized database architecture for the users module.
 */

export const genderEnum = pgEnum("gender", ["man", "woman", "non_binary"]);

export const users = pgTable(
  "users",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    // ==========================================
    // BETTER-AUTH REQUIRED FIELDS
    // ==========================================
    /** Display name (required by better-auth, auto-set from givenName + familyName) */
    name: text("name").notNull().default(""),

    // ==========================================
    // DEMOGRAPHIC DATA (OIDC standard naming)
    // ==========================================
    givenName: text("given_name"),
    familyName: text("family_name"),
    email: text("email").notNull().unique(),
    birthdate: date("birthdate"),
    gender: genderEnum("gender"),

    // ==========================================
    // AUTHENTICATION FIELDS (Better-Auth)
    // ==========================================
    emailVerified: boolean("email_verified").default(false).notNull(),
    image: text("image"),

    // ==========================================
    // TIMESTAMPS
    // ==========================================
    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [
    index("users_email_idx").on(table.email),
  ]
);

// ==========================================
// RELATIONS
// ==========================================

// Note: Relations with assessments and profiles are defined in their 
// respective schema files to prevent circular dependency issues.

// ==========================================
// INFERRED TYPES
// ==========================================

export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;
