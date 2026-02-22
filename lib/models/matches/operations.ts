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
 * Excludes users who already have a conversation with the current user.
 */

import { db } from "@/lib/db/drizzle";
import { users, type User } from "@/lib/models/users/schema";
import { embeddings } from "@/lib/models/embeddings/schema";
import { userInterests } from "@/lib/models/interests/schema";
import { members } from "@/lib/models/members/schema";
import { eventAttendees } from "@/lib/models/events/schema";
import { conversations } from "@/lib/models/conversations/schema";
import { eq, ne, and, or, sql, inArray, notExists } from "drizzle-orm";

// ─── Types ─────────────────────────────────────────────────────────

export interface MatchResult {
  user: {
    id: string;
    name: string;
    image: string | null;
    gender: string | null;
    birthdate: string | null;
  };
  score: number;
  distanceKm: number | null;
  sharedTags: string[];
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
  const myLocation = currentUser.location;
  if (!myLocation) return [];

  const myEmbeddingRow = await db.query.embeddings.findFirst({
    where: and(eq(embeddings.entityId, userId), eq(embeddings.entityType, "user")),
  });
  const myBehaviorEmbedding = myEmbeddingRow?.embedding ?? null;

  // Get current user's interest tags for overlap calculation
  const myInterests = await db.query.userInterests.findMany({
    where: eq(userInterests.userId, userId),
  });
  const myTags = myInterests.map((i) => i.tag);

  // Get current user's spaces and events
  const mySpaces = await db
    .select({ spaceId: members.spaceId })
    .from(members)
    .where(and(eq(members.userId, userId), eq(members.status, "active")));
  const mySpaceIds = mySpaces.map((s) => s.spaceId);

  const myEvents = await db
    .select({ eventId: eventAttendees.eventId })
    .from(eventAttendees)
    .where(
      and(
        eq(eventAttendees.userId, userId),
        inArray(eventAttendees.status, ["going", "attended"]),
      ),
    );
  const myEventIds = myEvents.map((e) => e.eventId);

  // Weighted interest similarity via user_interests table.
  // For each candidate: sum of min(myWeight, theirWeight) for shared tags,
  // divided by sum of max(myWeight, theirWeight) for the union of tags.
  // Falls back to 0 when either user has no interests.
  const interestScoreSql =
    myTags.length > 0
      ? sql<number>`COALESCE((
          SELECT SUM(LEAST(my.w, their.weight)) / NULLIF(
            SUM(LEAST(my.w, their.weight)) +
            (SELECT COALESCE(SUM(mi.weight), 0) FROM ${userInterests} mi WHERE mi.user_id = ${userId} AND mi.tag NOT IN (SELECT ui2.tag FROM ${userInterests} ui2 WHERE ui2.user_id = ${users.id})) +
            (SELECT COALESCE(SUM(ui3.weight), 0) FROM ${userInterests} ui3 WHERE ui3.user_id = ${users.id} AND ui3.tag NOT IN (SELECT mi2.tag FROM ${userInterests} mi2 WHERE mi2.user_id = ${userId}))
          , 0)
          FROM ${userInterests} their
          JOIN (VALUES ${sql.raw(myInterests.map((i) => `('${i.tag}'::text, ${i.weight}::real)`).join(","))}) AS my(tag, w)
            ON their.tag = my.tag
          WHERE their.user_id = ${users.id}
        ), 0)`
      : sql<number>`0`;

  const sharedSpacesCountSql = sql<number>`(
    SELECT count(*)::int 
    FROM ${members} m 
    WHERE m.user_id = ${users.id} 
      AND m.space_id = ANY(${mySpaceIds})
      AND m.status = 'active'
  )`;

  const sharedEventsCountSql = sql<number>`(
    SELECT count(*)::int 
    FROM ${eventAttendees} ea 
    WHERE ea.user_id = ${users.id} 
      AND ea.event_id = ANY(${myEventIds})
      AND ea.status IN ('going', 'attended')
  )`;

  const distanceSql = sql<number>`ST_DistanceSphere(${users.location}, ST_GeomFromText(${`POINT(${myLocation.x} ${myLocation.y})`}, 4326)) / 1000`;

  const embeddingStr = myBehaviorEmbedding ? `[${myBehaviorEmbedding.join(",")}]` : null;
  const behaviorSimSql = embeddingStr
    ? sql<number>`COALESCE((
        SELECT 1 - (e.embedding <=> ${embeddingStr}::vector)
        FROM ${embeddings} e
        WHERE e.entity_id = ${users.id} AND e.entity_type = 'user'
      ), 0)`
    : sql<number>`0`;

  const ageSql = sql<number>`date_part('year', age(${users.birthdate}))`;

  const spaceScoreSql =
    mySpaceIds.length > 0
      ? sql<number>`${sharedSpacesCountSql}::float / ${Math.max(mySpaceIds.length, 1)}`
      : sql<number>`0`;

  const eventScoreSql =
    myEventIds.length > 0
      ? sql<number>`${sharedEventsCountSql}::float / ${Math.max(myEventIds.length, 1)}`
      : sql<number>`0`;

  const proximityScoreSql = sql<number>`CASE WHEN ${distanceSql} <= ${filters.maxDistance} THEN 1 - (${distanceSql} / ${filters.maxDistance}) ELSE 0 END`;

  const finalScoreSql = sql<number>`(
    0.35 * ${interestScoreSql} +
    0.20 * ${spaceScoreSql} +
    0.20 * ${eventScoreSql} +
    0.15 * ${proximityScoreSql} +
    0.10 * ${behaviorSimSql}
  )`;

  // Shared tags for display
  const sharedTagsSql =
    myTags.length > 0
      ? sql<string[]>`ARRAY(
          SELECT ui.tag FROM ${userInterests} ui
          WHERE ui.user_id = ${users.id}
            AND ui.tag = ANY(${myTags}::text[])
        )`
      : sql<string[]>`'{}'::text[]`;

  const poolSize = filters.candidatePool ?? filters.limit;

  const candidates = await db
    .select({
      user: {
        id: users.id,
        name: users.name,
        image: users.image,
        gender: users.gender,
        birthdate: users.birthdate,
      },
      score: finalScoreSql,
      distanceKm: distanceSql,
      sharedTags: sharedTagsSql,
    })
    .from(users)
    .where(
      and(
        ne(users.id, userId),
        // Exclude users who already have a conversation with the current user
        notExists(
          db
            .select()
            .from(conversations)
            .where(
              or(
                and(
                  eq(conversations.initiatorId, userId),
                  eq(conversations.recipientId, users.id),
                ),
                and(
                  eq(conversations.initiatorId, users.id),
                  eq(conversations.recipientId, userId),
                ),
              ),
            ),
        ),
        // Candidates must have a location set
        sql`${users.location} IS NOT NULL`,
        // Must be within maxDistance radius
        sql`ST_DistanceSphere(${users.location}, ST_GeomFromText(${`POINT(${myLocation.x} ${myLocation.y})`}, 4326)) <= ${filters.maxDistance * 1000}`,
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
    )
    .orderBy(sql`${finalScoreSql} DESC`)
    .limit(poolSize)
    .offset(filters.offset);

  // If candidatePool was used, randomly sample `limit` from the pool
  const results = filters.candidatePool
    ? candidates.sort(() => Math.random() - 0.5).slice(0, filters.limit)
    : candidates;

  return results.map((c) => ({
    ...c,
    sharedSpaceIds: [],
    sharedEventIds: [],
  }));
}
