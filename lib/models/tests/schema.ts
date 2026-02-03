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
 * Schema per Test Sessions
 * 
 * Salva le risposte dell'utente al test.
 * Il test è identificato da un nome (es: "personality-v1").
 */

export const testStatusEnum = pgEnum("test_status", [
  "in_progress",
  "completed",
]);

export const testSessions = pgTable(
  "test_sessions",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    
    userId: uuid("user_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),
    
    /** Nome del test (es: "personality-v1") */
    testName: text("test_name").notNull(),
    
    /** 
     * Risposte al test
     * Record<questionId, valore>
     * - Chiuse: numero 1-5
     * - Aperte: stringa
     */
    answers: jsonb("answers").$type<TestAnswersJson>().notNull(),
    
    status: testStatusEnum("status").notNull().default("completed"),
    
    completedAt: timestamp("completed_at").defaultNow().notNull(),
  },
  (table) => [
    index("test_sessions_user_idx").on(table.userId),
    index("test_sessions_test_name_idx").on(table.testName),
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
export type TestAnswersJson = Record<string, number | string>;

// ============================================
// RELATIONS
// ============================================

export const testSessionsRelations = relations(testSessions, ({ one }) => ({
  user: one(users, {
    fields: [testSessions.userId],
    references: [users.id],
  }),
}));

// ============================================
// INFERRED TYPES
// ============================================

export type TestSession = typeof testSessions.$inferSelect;
export type NewTestSession = typeof testSessions.$inferInsert;
