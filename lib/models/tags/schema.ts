import {
    pgTable,
    text,
    timestamp,
} from "drizzle-orm/pg-core";
import { vector } from "drizzle-orm/pg-core/columns/vector_extension/vector";

export const tags = pgTable("tags", {
    id: text("id").primaryKey(), // We use the normalized tag name as the PK for simplicity and fast lookup
    name: text("name").notNull().unique(), // e.g. "digital_nomad"
    category: text("category").notNull(),  // e.g. "lifestyle"
    embedding: vector("embedding", { dimensions: 64 }).notNull(), // text-embedding-3-small pooled
    createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type Tag = typeof tags.$inferSelect;
export type NewTag = typeof tags.$inferInsert;
