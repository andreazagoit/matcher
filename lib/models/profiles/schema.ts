import {
  pgTable,
  uuid,
  timestamp,
  text,
  index,
  real,
} from "drizzle-orm/pg-core";
import { vector } from "drizzle-orm/pg-core/columns/vector_extension/vector";
import { relations } from "drizzle-orm";
import { users } from "@/lib/models/users/schema";

/**
 * User Profiles schema.
 * 
 * Each user has exactly one profile which contains textual descriptions
 * and corresponding vector embeddings for multi-axis matching.
 */

// Embedding dimensions (OpenAI text-embedding-3-small)
const EMBEDDING_DIMENSIONS = 1536;

export const profiles = pgTable(
  "profiles",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    /** 1:1 relation with users (UNIQUE) */
    userId: uuid("user_id")
      .notNull()
      .unique()
      .references(() => users.id, { onDelete: "cascade" }),

    // ==========================================
    // TEXTUAL DESCRIPTIONS
    // ==========================================

    /** PSYCHOLOGICAL axis description */
    psychologicalDesc: text("psychological_desc"),

    /** VALUES axis description */
    valuesDesc: text("values_desc"),

    /** INTERESTS axis description */
    interestsDesc: text("interests_desc"),

    /** BEHAVIORAL axis description */
    behavioralDesc: text("behavioral_desc"),

    // ==========================================
    // VECTOR EMBEDDINGS (ANN Search)
    // ==========================================

    /** PSYCHOLOGICAL - DOMINANT for ANN (weight 0.45) */
    psychologicalEmbedding: vector("psychological_embedding", {
      dimensions: EMBEDDING_DIMENSIONS
    }),

    /** VALUES (weight 0.25) */
    valuesEmbedding: vector("values_embedding", {
      dimensions: EMBEDDING_DIMENSIONS
    }),

    /** INTERESTS (weight 0.20) */
    interestsEmbedding: vector("interests_embedding", {
      dimensions: EMBEDDING_DIMENSIONS
    }),

    /** BEHAVIORAL (weight 0.10) */
    behavioralEmbedding: vector("behavioral_embedding", {
      dimensions: EMBEDDING_DIMENSIONS
    }),

    // ==========================================
    // METADATA
    // ==========================================

    /** Assessment version that generated this profile */
    assessmentVersion: real("assessment_version").default(1),

    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [
    // HNSW indices for ANN Search
    index("profiles_psychological_idx").using(
      "hnsw",
      table.psychologicalEmbedding.op("vector_cosine_ops")
    ),
    index("profiles_values_idx").using(
      "hnsw",
      table.valuesEmbedding.op("vector_cosine_ops")
    ),
    index("profiles_interests_idx").using(
      "hnsw",
      table.interestsEmbedding.op("vector_cosine_ops")
    ),
    index("profiles_behavioral_idx").using(
      "hnsw",
      table.behavioralEmbedding.op("vector_cosine_ops")
    ),

    // Index on userId
    index("profiles_user_idx").on(table.userId),
  ]
);

// ============================================
// RELATIONS
// ============================================

export const profilesRelations = relations(profiles, ({ one }) => ({
  user: one(users, {
    fields: [profiles.userId],
    references: [users.id],
  }),
}));

// ============================================
// INFERRED TYPES
// ============================================

export type Profile = typeof profiles.$inferSelect;
export type NewProfile = typeof profiles.$inferInsert;

// ============================================
// MATCHING CONSTANTS
// ============================================

/** Default weights used for weighted average result ranking. */
export const DEFAULT_MATCHING_WEIGHTS = {
  psychological: 0.45,
  values: 0.25,
  interests: 0.20,
  behavioral: 0.10,
} as const;
