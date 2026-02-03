import {
  pgTable,
  uuid,
  text,
  date,
  timestamp,
  pgEnum,
  index,
} from "drizzle-orm/pg-core";
import { vector } from "drizzle-orm/pg-core/columns/vector_extension/vector";
import { VALUES_OPTIONS } from "@/lib/models/values/operations";
import { INTERESTS_OPTIONS } from "@/lib/models/interests/operations";

// Enum PostgreSQL
export const valueEnum = pgEnum("value_enum", VALUES_OPTIONS);
export const interestEnum = pgEnum("interest_enum", INTERESTS_OPTIONS);

export const users = pgTable(
  "users",
  {
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
  },
  (table) => [
    // Indici HNSW per vector similarity search
    index("values_embedding_idx").using(
      "hnsw",
      table.valuesEmbedding.op("vector_cosine_ops")
    ),
    index("interests_embedding_idx").using(
      "hnsw",
      table.interestsEmbedding.op("vector_cosine_ops")
    ),
  ]
);

export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;

