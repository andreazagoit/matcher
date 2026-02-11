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
 * Assessment sessions and user responses.
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

    /** Assessment name (e.g., "personality-v1") */
    assessmentName: text("assessment_name").notNull(),

    /** 
     * Raw answers stored as a JSON object.
     * Record<questionId, string | number>
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
 * Dictionary of question IDs mapped to their respective answers.
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
