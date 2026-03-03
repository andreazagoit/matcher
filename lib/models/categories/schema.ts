import {
  pgTable,
  text,
  timestamp,
} from "drizzle-orm/pg-core";
import { vector } from "drizzle-orm/pg-core/columns/vector_extension/vector";

export const categories = pgTable("categories", {
  id: text("id").primaryKey(), // normalized name, e.g. "sport"
  name: text("name").notNull().unique(), // display name, e.g. "sport"
  embedding: vector("embedding", { dimensions: 64 }).notNull(), // text-embedding-3-small
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type Category = typeof categories.$inferSelect;
export type NewCategory = typeof categories.$inferInsert;
