/**
 * Match Operations
 *
 * Computes match scores between users based on:
 * 1. Weighted interest similarity (user_interests table)
 * 2. Shared Spaces
 * 3. Shared Events (co-attendance)
 * 4. Geographic proximity
 * 5. Behavioral similarity (cosine similarity of behavior embeddings)
 *
 * Excludes users who already have a connection with the current user.
 */

import { db } from "@/lib/db/drizzle";
import { users, type User } from "@/lib/models/users/schema";
import { embeddings } from "@/lib/models/embeddings/schema";
import { connections } from "@/lib/models/connections/schema";
import { eq, ne, and, or, sql, notExists, inArray } from "drizzle-orm";

// ─── Types ─────────────────────────────────────────────────────────

export interface MatchResult {
  user: {
    id: string;
    username: string;
    name: string;
    image: string | null;
    gender: string | null;
    birthdate: string;
  };
  score: number;
  distanceKm: number | null;
  sharedCategories: string[];
  sharedSpaceIds: string[];
  sharedEventIds: string[];
}

interface MatchFilters {
  maxDistance: number;
  limit: number;
  offset: number;
  candidatePool?: number; // fetch this many top candidates, then randomly sample `limit` from them
  gender?: NonNullable<User["gender"]>[];
  minAge?: number;
  maxAge?: number;
}

// ─── Find Matches ──────────────────────────────────────────────────

export async function findMatches(
  userId: string,
  filters: MatchFilters,
): Promise<MatchResult[]> {
  const currentUser = await db.query.users.findFirst({
    where: eq(users.id, userId),
  });

  if (!currentUser) throw new Error("User not found");

  // Location is required for daily matches
  const myLocation = currentUser.coordinates;
  if (!myLocation) return [];

  const myEmbeddingRow = await db.query.embeddings.findFirst({
    where: and(eq(embeddings.entityId, userId), eq(embeddings.entityType, "user")),
  });
  const myEmbedding = myEmbeddingRow?.embedding ?? null;

  const distanceSql = sql<number>`ST_DistanceSphere(${users.coordinates}, ST_GeomFromText(${`POINT(${myLocation.x} ${myLocation.y})`}, 4326)) / 1000`;

  const ageSql = sql<number>`date_part('year', age(${users.birthdate}))`;

  const embeddingStr = myEmbedding ? `[${myEmbedding.join(",")}]` : null;

  const poolSize = filters.candidatePool ?? filters.limit;

  const candidatesQuery = db
    .select({
      user: {
        id: users.id,
        username: users.username,
        name: users.name,
        image: users.image,
        gender: users.gender,
        birthdate: users.birthdate,
      },
      score: embeddingStr
        ? sql<number>`1 - (${embeddings.embedding} <=> ${sql.raw(`'${embeddingStr}'::vector`)})`
        : sql<number>`0`,
      distanceKm: distanceSql,
    })
    // Start from embeddings to use the HNSW index efficiently
    .from(embeddings)
    .innerJoin(users, eq(sql`${users.id}::text`, embeddings.entityId))
    .where(
      and(
        eq(embeddings.entityType, "user"),
        ne(users.id, userId),
        // Exclude ONLY accepted or declined connections
        notExists(
          db
            .select()
            .from(connections)
            .where(
              and(
                or(
                  and(
                    eq(connections.initiatorId, userId),
                    eq(connections.recipientId, users.id),
                  ),
                  and(
                    eq(connections.initiatorId, users.id),
                    eq(connections.recipientId, userId),
                  ),
                ),
                inArray(connections.status, ["accepted", "declined"])
              )
            ),
        ),
        // Candidates must have a location set
        sql`${users.coordinates} IS NOT NULL`,
        // Required by MatchUser GraphQL type
        sql`${users.username} IS NOT NULL`,
        sql`${users.birthdate} IS NOT NULL`,
        myLocation
          ? sql`ST_DistanceSphere(${users.coordinates}, ST_GeomFromText(${`POINT(${myLocation.x} ${myLocation.y})`}, 4326)) <= ${filters.maxDistance * 1000}`
          : undefined,
        filters.gender?.length
          ? inArray(users.gender, filters.gender)
          : undefined,
        filters.minAge
          ? sql`${ageSql} >= ${filters.minAge}`
          : undefined,
        filters.maxAge
          ? sql`${ageSql} <= ${filters.maxAge}`
          : undefined,
      ),
    );

  const candidates = await (embeddingStr
    ? candidatesQuery.orderBy(sql`${embeddings.embedding} <=> ${sql.raw(`'${embeddingStr}'::vector`)}`)
    : candidatesQuery.orderBy(sql`${distanceSql} ASC`)
  )
    .limit(poolSize)
    .offset(filters.offset);

  // If candidatePool was used, randomly sample `limit` from the pool
  const results = filters.candidatePool
    ? candidates.sort(() => Math.random() - 0.5).slice(0, filters.limit)
    : candidates;

  return results.map((c) => ({
    ...c,
    user: {
      ...c.user,
      username: c.user.username ?? "",
      birthdate: c.user.birthdate ?? "",
    },
    sharedCategories: [],
    sharedSpaceIds: [],
    sharedEventIds: [],
  }));
}
