import {
  pgTable,
  uuid,
  text,
  date,
  timestamp,
  pgEnum,
  customType,
} from "drizzle-orm/pg-core";
import { VALUES_OPTIONS, INTERESTS_OPTIONS } from "../constants";

// Custom type per pgvector (1536 = OpenAI embedding size)
const vector1536 = customType<{ data: number[]; driverData: string }>({
  dataType() {
    return "vector(1536)";
  },
  toDriver(value) {
    return JSON.stringify(value);
  },
  fromDriver(value) {
    if (typeof value === "string") {
      return JSON.parse(value);
    }
    return value as number[];
  },
});

// Enum PostgreSQL
export const valueEnum = pgEnum("value_enum", VALUES_OPTIONS);
export const interestEnum = pgEnum("interest_enum", INTERESTS_OPTIONS);

export const users = pgTable("users", {
  id: uuid("id").primaryKey().defaultRandom(),
  firstName: text("first_name").notNull(),
  lastName: text("last_name").notNull(),
  email: text("email").notNull().unique(),
  birthDate: date("birth_date").notNull(),
  values: valueEnum("values").array().notNull().default([]),
  interests: interestEnum("interests").array().notNull().default([]),
  valuesEmbedding: vector1536("values_embedding"),
  interestsEmbedding: vector1536("interests_embedding"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
  updatedAt: timestamp("updated_at").defaultNow().notNull(),
});

// SQL per creare gli indici HNSW (eseguire dopo db:push)
export const createIndexesSQL = `
CREATE INDEX IF NOT EXISTS values_embedding_idx ON users USING hnsw (values_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS interests_embedding_idx ON users USING hnsw (interests_embedding vector_cosine_ops);
`;

export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;
