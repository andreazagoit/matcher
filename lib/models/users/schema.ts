import {
  pgTable,
  uuid,
  text,
  date,
  timestamp,
  integer,
  index,
  pgEnum,
  boolean,
  geometry,
} from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";

/**
 * Normalized database architecture for the users module.
 */

export const genderEnum = pgEnum("gender", ["man", "woman", "non_binary"]);

export const sexualOrientationEnum = pgEnum("sexual_orientation", [
  "straight", "gay", "lesbian", "bisexual", "pansexual", "asexual", "queer", "other",
]);

export const relationshipIntentEnum = pgEnum("relationship_intent", [
  "serious_relationship", "casual_dating", "friendship", "chat",
]);

export const relationshipStyleEnum = pgEnum("relationship_style", [
  "monogamous", "ethical_non_monogamous", "open", "other",
]);

export const hasChildrenEnum = pgEnum("has_children", [
  "no", "yes",
]);

export const wantsChildrenEnum = pgEnum("wants_children", [
  "yes", "no", "open",
]);

export const religionEnum = pgEnum("religion", [
  "none", "christian", "muslim", "jewish", "buddhist", "hindu", "spiritual", "other",
]);

export const smokingEnum = pgEnum("smoking", [
  "never", "sometimes", "regularly",
]);

export const drinkingEnum = pgEnum("drinking", [
  "never", "sometimes", "regularly",
]);

export const activityLevelEnum = pgEnum("activity_level", [
  "sedentary", "light", "moderate", "active", "very_active",
]);

export const educationLevelEnum = pgEnum("education_level", [
  "middle_school", "high_school", "bachelor", "master", "phd", "vocational", "other",
]);

export const ethnicityEnum = pgEnum("ethnicity", [
  "white_caucasian", "hispanic_latino", "black_african", "east_asian",
  "south_asian", "middle_eastern", "pacific_islander", "indigenous", "mixed", "other",
]);

export const users = pgTable(
  "users",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    // ==========================================
    // BETTER-AUTH REQUIRED FIELDS
    // ==========================================
    /** Display name (required by better-auth) */
    name: text("name").notNull().default(""),

    // ==========================================
    // DEMOGRAPHIC DATA (OIDC standard naming)
    // ==========================================
    username: text("username").unique(),
    email: text("email").notNull().unique(),
    birthdate: date("birthdate"),
    gender: genderEnum("gender"),

    // ==========================================
    // PROFILE — ORIENTATION & IDENTITY
    // ==========================================
    sexualOrientation: text("sexual_orientation").array().default(sql`'{}'::text[]`).notNull(),
    heightCm: integer("height_cm"),

    // ==========================================
    // PROFILE — RELATIONAL INTENT
    // ==========================================
    relationshipIntent: text("relationship_intent").array().default(sql`'{}'::text[]`).notNull(),
    relationshipStyle: relationshipStyleEnum("relationship_style"),
    hasChildren: hasChildrenEnum("has_children"),
    wantsChildren: wantsChildrenEnum("wants_children"),

    // ==========================================
    // PROFILE — LIFESTYLE
    // ==========================================
    religion: religionEnum("religion"),
    smoking: smokingEnum("smoking"),
    drinking: drinkingEnum("drinking"),
    activityLevel: activityLevelEnum("activity_level"),

    // ==========================================
    // PROFILE — INTERESTS
    // ==========================================
    tags: text("tags").array().default(sql`'{}'::text[]`).notNull(),

    // ==========================================
    // PROFILE — IDENTITY & BACKGROUND
    // ==========================================
    jobTitle: text("job_title"),
    educationLevel: educationLevelEnum("education_level"),
    schoolName: text("school_name"),
    languages: text("languages").array().default(sql`'{}'::text[]`).notNull(),
    ethnicity: ethnicityEnum("ethnicity"),

    // ==========================================
    // AUTHENTICATION FIELDS (Better-Auth)
    // ==========================================
    emailVerified: boolean("email_verified").default(false).notNull(),
    image: text("image"),

    // ==========================================
    // LOCATION (PostGIS)
    // x = longitude, y = latitude (PostGIS convention)
    // ==========================================
    locationText: text("location_text"),
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
