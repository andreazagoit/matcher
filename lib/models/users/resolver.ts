/**
 * User resolvers — CRUD and location.
 */

import {
  createUser,
  updateUser,
  deleteUser,
  getUserByUsername,
  getAllUsers,
  updateUserLocation,
  isUsernameTaken,
} from "./operations";
import { users } from "./schema";
import type { CreateUserInput, UpdateUserInput } from "./validator";
import type { GraphQLContext } from "@/lib/graphql/context";
import {
  embedUser,
  recommendEventsForUser,
  recommendSpacesForUser,
} from "@/lib/models/embeddings/operations";
import { embeddings } from "@/lib/models/embeddings/schema";
import { db } from "@/lib/db/drizzle";
import { eq, inArray, and, gte, sql } from "drizzle-orm";
import { isValidCategory as _isValidCategory } from "@/lib/models/categories/data"; // kept for potential future use
import { getUserItems } from "@/lib/models/useritems/operations";
import type { UserItem } from "@/lib/models/useritems/schema";
import { events, eventAttendees } from "@/lib/models/events/schema";
import { spaces } from "@/lib/models/spaces/schema";
import { members } from "@/lib/models/members/schema";
import { GraphQLError } from "graphql";

class AuthError extends Error {
  constructor(message: string, public code: string = "UNAUTHENTICATED") {
    super(message);
    this.name = "AuthError";
  }
}

export const userResolvers = {
  Query: {
    me: async (_: unknown, __: unknown, context: GraphQLContext) => {
      if (!context.auth.user) {
        return null;
      }
      return context.auth.user;
    },

    user: async (
      _: unknown,
      { username }: { username: string },
    ) => {
      return await getUserByUsername(username);
    },

    users: async (_: unknown, __: unknown, context: GraphQLContext) => {
      if (!context.auth.user) {
        throw new AuthError("Authentication required");
      }
      return await getAllUsers();
    },

    checkUsername: async (_: unknown, { username }: { username: string }) => {
      return await isUsernameTaken(username);
    },
  },

  Mutation: {
    createUser: async (
      _: unknown,
      { input }: { input: CreateUserInput },
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) {
        throw new AuthError("Authentication required");
      }
      return await createUser(input);
    },

    updateUser: async (
      _: unknown,
      { id, input }: { id: string; input: UpdateUserInput },
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) {
        throw new AuthError("Authentication required");
      }

      // Users can only update their own profile
      if (context.auth.user.id !== id) {
        throw new AuthError("You can only update your own profile", "FORBIDDEN");
      }

      const updatedUser = await updateUser(id, input);

      // Regenerate embedding in background when profile data changes
      (async () => {
        await embedUser(id, {
          birthdate: updatedUser.birthdate ?? null,
          gender: updatedUser.gender ?? null,
          relationshipIntent: updatedUser.relationshipIntent ?? null,
          smoking: updatedUser.smoking ?? null,
          drinking: updatedUser.drinking ?? null,
          activityLevel: updatedUser.activityLevel ?? null,
        });
      })().catch(() => { });

      return updatedUser;
    },

    deleteUser: async (
      _: unknown,
      { id }: { id: string },
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) {
        throw new AuthError("Authentication required");
      }

      if (context.auth.user.id !== id) {
        throw new AuthError("Forbidden", "FORBIDDEN");
      }

      return await deleteUser(id);
    },

    updateLocation: async (
      _: unknown,
      { lat, lon }: { lat: number; lon: number; location?: string },
      context: GraphQLContext,
    ) => {
      if (!context.auth.user) {
        throw new AuthError("Authentication required");
      }

      return await updateUserLocation(context.auth.user.id, lat, lon);
    },
  },

  User: {
    coordinates: (parent: { coordinates?: { x: number; y: number } | null }) => {
      if (!parent.coordinates) return null;
      return { lat: parent.coordinates.y, lon: parent.coordinates.x };
    },
    userItems: (parent: { id: string }): Promise<UserItem[]> => {
      return getUserItems(parent.id);
    },

    recommendedCategories: async (
      parent: { id: string },
      { limit = 10, offset = 0 }: { limit?: number; offset?: number },
      context: GraphQLContext,
    ) => {
      if (context.auth.user?.id !== parent.id) return [];
      const row = await db.query.embeddings.findFirst({
        where: and(
          eq(embeddings.entityId, parent.id),
          eq(embeddings.entityType, "user"),
        ),
        columns: { embedding: true },
      });
      if (!row) return [];
      const vec = `[${row.embedding.join(",")}]`;
      const rows = await db.execute<{ entity_id: string }>(sql`
        SELECT entity_id
        FROM   embeddings
        WHERE  entity_type = 'category'
        ORDER BY embedding <=> ${sql.raw(`'${vec}'::vector`)}
        LIMIT  ${sql.raw(String(limit))}
        OFFSET ${sql.raw(String(offset))}
      `);
      return rows.map((r) => r.entity_id);
    },

    recommendedUserUsers: async (
      parent: { id: string },
      { limit = 10, offset = 0 }: { limit?: number; offset?: number },
      context: GraphQLContext,
    ) => {
      if (context.auth.user?.id !== parent.id) return [];
      const row = await db.query.embeddings.findFirst({
        where: and(
          eq(embeddings.entityId, parent.id),
          eq(embeddings.entityType, "user"),
        ),
        columns: { embedding: true },
      });
      if (!row) return [];
      const vec = `[${row.embedding.join(",")}]`;
      const emb = await db.execute<{ entity_id: string }>(sql`
        SELECT entity_id
        FROM   embeddings
        WHERE  entity_type = 'user'
          AND  entity_id   != ${parent.id}
        ORDER BY embedding <=> ${sql.raw(`'${vec}'::vector`)}
        LIMIT  ${sql.raw(String(limit))}
        OFFSET ${sql.raw(String(offset))}
      `);
      const ids = emb.map((r) => r.entity_id);
      if (!ids.length) return [];
      const rows = await db.select().from(users).where(inArray(users.id, ids));
      const map = new Map(rows.map((u) => [u.id, u]));
      return ids.map((id) => map.get(id)).filter(Boolean);
    },

    recommendedEvents: async (
      parent: { id: string },
      { limit = 10, offset = 0 }: { limit?: number; offset?: number },
      context: GraphQLContext,
    ) => {
      if (context.auth.user?.id !== parent.id) return [];

      // Exclude events already attended
      const attended = await db
        .select({ eventId: eventAttendees.eventId })
        .from(eventAttendees)
        .where(eq(eventAttendees.userId, parent.id));
      const excludeIds = attended.map((a) => a.eventId);

      const ids = await recommendEventsForUser(parent.id, limit, offset, excludeIds);
      if (!ids.length) return [];

      const rows = await db
        .select()
        .from(events)
        .where(
          and(
            inArray(events.id, ids),
            gte(events.startsAt, new Date()),
          ),
        );
      const map = new Map(rows.map((e) => [e.id, e]));
      return ids.map((id) => map.get(id)).filter(Boolean);
    },

    recommendedSpaces: async (
      parent: { id: string },
      { limit = 10, offset = 0 }: { limit?: number; offset?: number },
      context: GraphQLContext,
    ) => {
      if (context.auth.user?.id !== parent.id) return [];

      // Exclude spaces user already joined
      const joined = await db
        .select({ spaceId: members.spaceId })
        .from(members)
        .where(and(eq(members.userId, parent.id), eq(members.status, "active")));
      const excludeIds = joined.map((m) => m.spaceId);

      const ids = await recommendSpacesForUser(parent.id, limit, offset, excludeIds);
      if (!ids.length) return [];

      const rows = await db
        .select()
        .from(spaces)
        .where(
          and(
            inArray(spaces.id, ids),
            eq(spaces.visibility, "public"),
            eq(spaces.isActive, true),
          ),
        );
      const map = new Map(rows.map((s) => [s.id, s]));
      return ids.map((id) => map.get(id)).filter(Boolean);
    },
  },
};
