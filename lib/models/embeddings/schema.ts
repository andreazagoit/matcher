import {
  pgTable,
  uuid,
  text,
  timestamp,
  index,
  uniqueIndex,
} from "drizzle-orm/pg-core";
import { vector } from "drizzle-orm/pg-core/columns/vector_extension/vector";

/**
 * Embeddings — 256-dim vectors for cross-entity recommendations.
 *
 * Stores a comparable embedding for every entity (user, event, space, tag)
 * in a shared vector space. All entity types can be compared with each other
 * via cosine similarity using pgvector (HNSW index for ANN search).
 *
 * entityId is text (not uuid) to support tag entities, which use their
 * string name (e.g. "music", "travel") as identifier rather than a UUID.
 */

export const EMBEDDING_DIMENSIONS = 256;

export const entityTypeEnum = ["user", "event", "space", "tag"] as const;

export const embeddings = pgTable(
  "embeddings",
  {
    id: uuid("id").primaryKey().defaultRandom(),

    entityId: text("entity_id").notNull(),

    entityType: text("entity_type", { enum: entityTypeEnum }).notNull(),

    embedding: vector("embedding", { dimensions: EMBEDDING_DIMENSIONS }).notNull(),

    updatedAt: timestamp("updated_at").defaultNow().notNull(),
  },
  (table) => [
    uniqueIndex("embeddings_entity_idx").on(table.entityId, table.entityType),
    index("embeddings_type_idx").on(table.entityType),
    // HNSW index for fast approximate nearest neighbor search
    // Allows O(log N) similarity queries instead of O(N) full scan
    index("embeddings_hnsw_idx").using(
      "hnsw",
      table.embedding.op("vector_cosine_ops"),
    ),
  ],
);

export type Embedding = typeof embeddings.$inferSelect;
export type NewEmbedding = typeof embeddings.$inferInsert;
