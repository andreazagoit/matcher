import {
  pgTable,
  uuid,
  text,
  timestamp,
  jsonb,
  pgEnum,
  index,
} from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";
import { users } from "@/lib/models/users/schema";

/**
 * Schema per Assessments
 * 
 * Salva le risposte dell'utente all'assessment.
 * L'assessment è identificato da un nome (es: "personality-v1").
 */

export const assessmentStatusEnum = pgEnum("assessment_status", [
  "in_progress",
  "completed",
]);

export const assessments = pgTable(
  "assessments",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    userId: uuid("user_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),

    /** Nome dell'assessment (es: "personality-v1") */
    assessmentName: text("assessment_name").notNull(),

    /** 
     * Risposte al test
     * Record<questionId, valore>
     * - Chiuse: numero 1-5
     * - Aperte: stringa
     */
    answers: jsonb("answers").$type<AssessmentAnswersJson>().notNull(),

    status: assessmentStatusEnum("status").notNull().default("completed"),

    completedAt: timestamp("completed_at").defaultNow().notNull(),
  },
  (table) => [
    index("assessments_user_idx").on(table.userId),
    index("assessments_assessment_name_idx").on(table.assessmentName),
  ]
);

// ============================================
// TYPES
// ============================================

/**
 * Formato risposte: { questionId: valore }
 * - Chiuse: valore 1-5 (intero)
 * - Aperte: stringa
 */
export type AssessmentAnswersJson = Record<string, number | string>;

// ============================================
// RELATIONS
// ============================================

export const assessmentsRelations = relations(assessments, ({ one }) => ({
  user: one(users, {
    fields: [assessments.userId],
    references: [users.id],
  }),
}));

// ============================================
// INFERRED TYPES
// ============================================

export type Assessment = typeof assessments.$inferSelect;
export type NewAssessment = typeof assessments.$inferInsert;
