import {
  pgTable,
  uuid,
  text,
  date,
  timestamp,
  index,
  pgEnum,
  boolean,
  real,
  integer,
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
    // DEMOGRAPHIC DATA
    // ==========================================
    firstName: text("first_name").notNull(),
    lastName: text("last_name").notNull(),
    email: text("email").notNull().unique(),
    birthDate: date("birth_date").notNull(),
    gender: genderEnum("gender"),

    // ==========================================
    // LOCALE & GEOLOCATION
    // ==========================================
    /** Spoken languages (ISO 639-1 codes, e.g. ['it', 'en']) */
    languages: text("languages").array().default([]).notNull(),
    /** GPS latitude (nullable — user may decline) */
    latitude: real("latitude"),
    /** GPS longitude (nullable — user may decline) */
    longitude: real("longitude"),

    // ==========================================
    // AUTHENTICATION FIELDS (NextAuth/Better-Auth)
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
