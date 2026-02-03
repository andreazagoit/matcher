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

export type User = typeof users.$inferSelect;
export type NewUser = typeof users.$inferInsert;
