import {
  pgTable,
  uuid,
  text,
  date,
  timestamp,
  index,
  pgEnum,
} from "drizzle-orm/pg-core";

/**
 * ARCHITETTURA NORMALIZZATA:
 * - users: dati base (questa tabella)
 * - assessments: sessioni di assessment
 * - assessment_answers: risposte agli assessment (JSON in assessments)
 * - profiles: profilo calcolato + embeddings (per matching)
 * 
 * Vantaggi:
 * - Tabella users snella
 * - Storico assessment completo
 * - Versioning questionari
 * - Confronto risposte raw tra utenti
 */

export const genderEnum = pgEnum("gender", ["man", "woman", "non_binary"]);

export const users = pgTable(
  "users",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    // ==========================================
    // DATI ANAGRAFICI
    // ==========================================
    firstName: text("first_name").notNull(),
    lastName: text("last_name").notNull(),
    email: text("email").notNull().unique(),
    birthDate: date("birth_date").notNull(),
    gender: genderEnum("gender"),

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
// RELATIONS (definite qui, riferimenti lazy)
// ==========================================

// Le relazioni con assessments e profiles sono definite
// nei rispettivi file schema per evitare circular imports

// ==========================================
// TIPI INFERITI
// ==========================================

export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;
