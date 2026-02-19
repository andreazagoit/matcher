import {
  pgTable,
  uuid,
  text,
  date,
  timestamp,
  index,
  pgEnum,
  boolean,
  geometry,
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
    username: text("username").unique(),
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
    // LOCATION (PostGIS)
    // x = longitude, y = latitude (PostGIS convention)
    // ==========================================
    location: geometry("location", { type: "point", mode: "xy", srid: 4326 }),
    locationUpdatedAt: timestamp("location_updated_at"),

    // ==========================================
    // TIMESTAMPS
    // ==========================================
    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [
    index("users_email_idx").on(table.email),
    index("users_username_idx").on(table.username),
    index("users_location_gist_idx").using("gist", table.location),
  ]
);

// ==========================================
// RELATIONS
// ==========================================


// ==========================================
// INFERRED TYPES
// ==========================================

export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;
