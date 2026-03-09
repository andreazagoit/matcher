/**
 * User resolvers — CRUD, location, and personalised recommendations.
 * Recommended* resolvers live at Query root (not on User type) so they are
 * only executed when explicitly requested, not whenever a User fragment is used.
 */

import {
  updateUser,
  deleteUser,
  getUserByUsername,
  updateUserLocation,
  isUsernameTaken,
  getUserRecommendedEvents,
} from "./operations";
import type { UpdateUserInput } from "./validator";
import type { GraphQLContext } from "@/lib/graphql/context";
import { GraphQLError } from "graphql";
import { embedUser, recommendSpacesForUser } from "@/lib/models/embeddings/operations";
import { db } from "@/lib/db/drizzle";
import { users } from "./schema";
import { spaces } from "@/lib/models/spaces/schema";
import { members } from "@/lib/models/members/schema";
import { embeddings } from "@/lib/models/embeddings/schema";
import { eq, and, inArray, sql } from "drizzle-orm";
import { getAllEvents } from "@/lib/models/events/operations";
import { getAllSpaces } from "@/lib/models/spaces/operations";

const EMPTY_SPACE_CONNECTION = { nodes: [], hasNextPage: false };

export const userResolvers = {
  Query: {
    me: async (_: unknown, __: unknown, context: GraphQLContext) => {
      return context.auth.user ?? null;
    },

    user: async (_: unknown, { username }: { username: string }) => {
      return getUserByUsername(username);
    },

    checkUsername: async (_: unknown, { username }: { username: string }) => {
      return isUsernameTaken(username);
    },

    recommendedEvents: async (
      _: unknown,
      { limit = 10, offset = 0 }: { limit?: number; offset?: number },
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) {
        const rows = await getAllEvents(limit + 1, offset);
        return { nodes: rows.slice(0, limit), hasNextPage: rows.length > limit };
      }
      const userId = context.auth.user.id;
      const nodes = await getUserRecommendedEvents(userId, limit + 1, offset);
      return { nodes: nodes.slice(0, limit), hasNextPage: nodes.length > limit };
    },

    recommendedSpaces: async (
      _: unknown,
      { limit = 10, offset = 0 }: { limit?: number; offset?: number },
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) {
        const rows = await getAllSpaces(limit + 1, offset);
        return { nodes: rows.slice(0, limit), hasNextPage: rows.length > limit };
      }
      const userId = context.auth.user.id;
      const joined = await db
        .select({ spaceId: members.spaceId })
        .from(members)
        .where(and(eq(members.userId, userId), eq(members.status, "active")));
      const excludeIds = joined.map((m) => m.spaceId);
      const ids = await recommendSpacesForUser(userId, limit + 1, offset, excludeIds);
      if (!ids.length) return EMPTY_SPACE_CONNECTION;
      const rows = await db
        .select()
        .from(spaces)
        .where(and(inArray(spaces.id, ids), eq(spaces.visibility, "public")));
      const map = new Map(rows.map((s) => [s.id, s]));
      const nodes = ids.map((id) => map.get(id)).filter(Boolean);
      return { nodes: nodes.slice(0, limit), hasNextPage: nodes.length > limit };
    },

    recommendedUsers: async (
      _: unknown,
      { limit = 10, offset = 0 }: { limit?: number; offset?: number },
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) return [];
      const userId = context.auth.user.id;
      const row = await db.query.embeddings.findFirst({
        where: and(eq(embeddings.entityId, userId), eq(embeddings.entityType, "user")),
        columns: { embedding: true },
      });
      if (!row) return [];
      const vec = `[${row.embedding.join(",")}]`;
      const emb = await db.execute<{ entity_id: string }>(sql`
        SELECT entity_id FROM embeddings
        WHERE entity_type = 'user' AND entity_id != ${userId}
        ORDER BY embedding <=> ${sql.raw(`'${vec}'::vector`)}
        LIMIT ${sql.raw(String(limit))} OFFSET ${sql.raw(String(offset))}
      `);
      const ids = emb.map((r) => r.entity_id);
      if (!ids.length) return [];
      const userRows = await db.select().from(users).where(inArray(users.id, ids));
      const map = new Map(userRows.map((u) => [u.id, u]));
      return ids.map((id) => map.get(id)).filter(Boolean);
    },

    recommendedCategories: async (
      _: unknown,
      { limit = 10, offset = 0 }: { limit?: number; offset?: number },
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) return [];
      const userId = context.auth.user.id;
      const row = await db.query.embeddings.findFirst({
        where: and(eq(embeddings.entityId, userId), eq(embeddings.entityType, "user")),
        columns: { embedding: true },
      });
      if (!row) return [];
      const vec = `[${row.embedding.join(",")}]`;
      const rows = await db.execute<{ entity_id: string }>(sql`
        SELECT entity_id FROM embeddings
        WHERE entity_type = 'category'
        ORDER BY embedding <=> ${sql.raw(`'${vec}'::vector`)}
        LIMIT ${sql.raw(String(limit))} OFFSET ${sql.raw(String(offset))}
      `);
      return rows.map((r) => ({ id: r.entity_id }));
    },
  },

  Mutation: {
    updateUser: async (
      _: unknown,
      { id, input }: { id: string; input: UpdateUserInput },
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) throw new GraphQLError("Authentication required", { extensions: { code: "UNAUTHENTICATED" } });
      if (context.auth.user.id !== id) throw new GraphQLError("You can only update your own profile", { extensions: { code: "FORBIDDEN" } });

      const updatedUser = await updateUser(id, input);

      (async () => {
        await embedUser(id, {
          birthdate: updatedUser.birthdate ?? null,
          gender: updatedUser.gender ?? null,
          relationshipIntent: updatedUser.relationshipIntent ?? null,
          smoking: updatedUser.smoking ?? null,
          drinking: updatedUser.drinking ?? null,
          activityLevel: updatedUser.activityLevel ?? null,
        });
      })().catch(() => {});

      return updatedUser;
    },

    deleteUser: async (_: unknown, { id }: { id: string }, context: GraphQLContext) => {
      if (!context.auth.user) throw new GraphQLError("Authentication required", { extensions: { code: "UNAUTHENTICATED" } });
      if (context.auth.user.id !== id) throw new GraphQLError("Forbidden", { extensions: { code: "FORBIDDEN" } });
      return deleteUser(id);
    },

    updateLocation: async (
      _: unknown,
      { lat, lon, location }: { lat: number; lon: number; location?: string },
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) throw new GraphQLError("Authentication required", { extensions: { code: "UNAUTHENTICATED" } });
      return updateUserLocation(context.auth.user.id, lat, lon, location);
    },
  },

  User: {
    coordinates: (parent: { coordinates?: { x: number; y: number } | null }) => {
      if (!parent.coordinates) return null;
      return { lat: parent.coordinates.y, lon: parent.coordinates.x };
    },
  },
};
