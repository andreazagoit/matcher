import {
  pgTable,
  uuid,
  text,
  date,
  timestamp,
  pgEnum,
} from "drizzle-orm/pg-core";
import { vector } from "drizzle-orm/pg-core/columns/vector_extension/vector";
import { VALUES_OPTIONS, INTERESTS_OPTIONS } from "../constants";

// Enum PostgreSQL
export const valueEnum = pgEnum("value_enum", VALUES_OPTIONS);
export const interestEnum = pgEnum("interest_enum", INTERESTS_OPTIONS);

export const users = pgTable("users", {
  id: uuid("id").primaryKey().defaultRandom(),
  firstName: text("first_name").notNull(),
  lastName: text("last_name").notNull(),
  email: text("email").notNull().unique(),
  birthDate: date("birth_date").notNull(),
  // Obbligatori, senza default
  values: valueEnum("values").array().notNull(),
  interests: interestEnum("interests").array().notNull(),
  // Embeddings obbligatori (1536 = OpenAI embedding size)
  valuesEmbedding: vector("values_embedding", { dimensions: 1536 }).notNull(),
  interestsEmbedding: vector("interests_embedding", { dimensions: 1536 }).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;

