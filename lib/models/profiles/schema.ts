import {
  pgTable,
  uuid,
  timestamp,
  jsonb,
  index,
  real,
} from "drizzle-orm/pg-core";
import { vector } from "drizzle-orm/pg-core/columns/vector_extension/vector";
import { relations } from "drizzle-orm";
import { users } from "@/lib/models/users/schema";

/**
 * User Profiles - Profilo Utente Standard (SEMPLIFICATO)
 * 
 * PRINCIPI:
 * 1. Un profilo per utente (1:1)
 * 2. Struttura STANDARD indipendente dalla versione dell'assessment
 * 3. Retrocompatibile: qualsiasi versione dell'assessment genera lo stesso formato
 * 
 * Il profilo contiene:
 * - 4 assi di traits (psychological, values, interests, behavioral)
 * - 4 embeddings per ANN matching
 * - Ogni asse ha: traits normalizzati (0-1) + descrizione testuale
 */

// Dimensione embedding (OpenAI text-embedding-3-small = 1536)
const EMBEDDING_DIMENSIONS = 1536;

export const profiles = pgTable(
  "profiles",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    /** Relazione 1:1 con users (UNIQUE) */
    userId: uuid("user_id")
      .notNull()
      .unique()
      .references(() => users.id, { onDelete: "cascade" }),

    // ==========================================
    // TRAITS STANDARD (struttura fissa, retrocompatibile)
    // ==========================================

    /**
     * Asse PSYCHOLOGICAL (dominante per ANN)
     * Traits: extraversion, openness, conscientiousness, agreeableness,
     *         emotionalStability, empathy, impulsivity
     */
    psychological: jsonb("psychological").$type<ProfileAxis>(),

    /**
     * Asse VALUES
     * Traits: honesty, loyalty, workLifeBalance, independence, tradition, progressivism
     */
    values: jsonb("values").$type<ProfileAxis>(),

    /**
     * Asse INTERESTS
     * Traits: activeLifestyle, socialActivities, culturalInterests
     * + hobbies list
     */
    interests: jsonb("interests").$type<ProfileAxis>(),

    /**
     * Asse BEHAVIORAL
     * Traits: responsiveness, communicationDepth, interactionFrequency
     */
    behavioral: jsonb("behavioral").$type<ProfileAxis>(),

    // ==========================================
    // 4 EMBEDDINGS (per ANN Search)
    // ==========================================

    /** PSYCHOLOGICAL - DOMINANTE per ANN (peso 0.45) */
    psychologicalEmbedding: vector("psychological_embedding", {
      dimensions: EMBEDDING_DIMENSIONS
    }),

    /** VALUES (peso 0.25) */
    valuesEmbedding: vector("values_embedding", {
      dimensions: EMBEDDING_DIMENSIONS
    }),

    /** INTERESTS (peso 0.20) */
    interestsEmbedding: vector("interests_embedding", {
      dimensions: EMBEDDING_DIMENSIONS
    }),

    /** BEHAVIORAL (peso 0.10) */
    behavioralEmbedding: vector("behavioral_embedding", {
      dimensions: EMBEDDING_DIMENSIONS
    }),

    // ==========================================
    // METADATA
    // ==========================================

    /** Versione dell'assessment che ha generato questo profilo */
    assessmentVersion: real("assessment_version").default(1),

    /** Timestamps */
    createdAt: timestamp("created_at").defaultNow().notNull(),
    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [
    // Indici HNSW per ANN Search
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

    // Indice su userId
    index("profiles_user_idx").on(table.userId),
  ]
);

// ============================================
// TYPES STANDARD (retrocompatibili)
// ============================================

/**
 * Struttura standard per ogni asse del profilo
 * Questa struttura NON cambia tra versioni del test
 */
export interface ProfileAxis {
  /** Traits normalizzati 0-1 */
  traits: Record<string, number>;

  /** Descrizione testuale (input per embedding) */
  description: string;

  /** Dati extra (es. lista hobby per profiles) */
  extra?: Record<string, unknown>;
}

/**
 * Traits standard per ogni asse
 * Queste chiavi sono FISSE e retrocompatibili
 */
export const STANDARD_TRAITS = {
  psychological: [
    "extraversion",
    "openness",
    "conscientiousness",
    "agreeableness",
    "emotionalStability",
    "empathy",
    "impulsivity",
  ],
  values: [
    "honesty",
    "loyalty",
    "workLifeBalance",
    "independence",
    "tradition",
    "progressivism",
  ],
  interests: [
    "activeLifestyle",
    "socialActivities",
    "culturalInterests",
  ],
  behavioral: [
    "responsiveness",
    "communicationDepth",
    "interactionFrequency",
  ],
} as const;

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
// COSTANTI MATCHING
// ============================================

/** Pesi default per il ranking finale */
export const DEFAULT_MATCHING_WEIGHTS = {
  psychological: 0.45,
  values: 0.25,
  interests: 0.20,
  behavioral: 0.10,
} as const;
