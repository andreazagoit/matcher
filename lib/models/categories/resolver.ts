import { getCategories, getCategoryById, createCategory } from "./operations";
import { db } from "@/lib/db/drizzle";
import { embeddings } from "@/lib/models/embeddings/schema";
import { events } from "@/lib/models/events/schema";
import { spaces } from "@/lib/models/spaces/schema";
import { eq, and, sql, gte } from "drizzle-orm";

export const categoryResolvers = {
  Query: {
    categories: async () => {
      const rows = await getCategories();
      const unique = [...new Set(rows.map((r) => r.id))];
      return unique.map((id) => ({ id }));
    },

    category: async (_: unknown, { id }: { id: string }) => {
      const row = await getCategoryById(id);
      return row ? { id: row.id } : null;
    },
  },

  Mutation: {
    createCategory: async (_: unknown, { name }: { name: string }) => {
      return await createCategory(name);
    },
  },

  Category: {
    recommendedEvents: async (
      parent: { id: string },
      { limit = 20, offset = 0 }: { limit?: number; offset?: number },
    ) => {
      const catArray = `{${parent.id}}`;
      return await db
        .select()
        .from(events)
        .where(
          and(
            sql`${events.categories} && ${catArray}::text[]`,
            gte(events.startsAt, new Date()),
          ),
        )
        .orderBy(sql`${events.startsAt} ASC`)
        .limit(limit)
        .offset(offset);
    },

    recommendedSpaces: async (
      parent: { id: string },
      { limit = 20, offset = 0 }: { limit?: number; offset?: number },
    ) => {
      const catArray = `{${parent.id}}`;
      return await db
        .select()
        .from(spaces)
        .where(
          and(
            sql`${spaces.categories} && ${catArray}::text[]`,
            eq(spaces.visibility, "public"),
            eq(spaces.isActive, true),
          ),
        )
        .limit(limit)
        .offset(offset);
    },

    recommendedCategories: async (
      parent: { id: string },
      { limit = 6 }: { limit?: number },
    ) => {
      const row = await db.query.embeddings.findFirst({
        where: and(
          eq(embeddings.entityId, parent.id),
          eq(embeddings.entityType, "category"),
        ),
        columns: { embedding: true },
      });
      if (!row) return [];

      const vec = `[${row.embedding.join(",")}]`;
      const rows = await db.execute<{ entity_id: string }>(sql`
        SELECT entity_id
        FROM   embeddings
        WHERE  entity_type = 'category'
          AND  entity_id   != ${parent.id}
        ORDER BY embedding <=> ${sql.raw(`'${vec}'::vector`)}
        LIMIT  ${sql.raw(String(limit))}
      `);
      return rows.map((r) => r.entity_id);
    },
  },
};
