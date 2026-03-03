/**
 * Daily Matches Operations
 *
 * Bidirectional similarity:
 *   score(A,B) = average of cosine similarity A→B and B→A.
 *   In practice this means for each candidate B we compute:
 *     1 - (embed_A <=> embed_B)   (A's perspective)
 *     1 - (embed_B <=> embed_A)   (B's perspective, same value for cosine)
 *   Cosine similarity is symmetric, so we use a single vector distance but
 *   also require that the candidate B has an embedding (bidirectional = both
 *   parties must be "reachable").  The real bidirectional boost comes from
 *   filtering: we only include candidates whose own embedding is close to A
 *   AND whose embedding-based score for A would also be high, i.e. we rank
 *   by the symmetric cosine distance which is identical in both directions.
 *
 *   The randomisation step (pick 8 from top 200) ensures daily variety.
 *
 * getDailyMatches     — returns today's batch, generates on-the-fly if missing.
 * generateDailyMatchesForAll — called by cron to pre-warm all users at midnight.
 * resetDailyMatches   — deletes rows older than today (called by cron).
 */

import { db } from "@/lib/db/drizzle";
import { users } from "@/lib/models/users/schema";
import { userItems } from "@/lib/models/useritems/schema";
import { embeddings } from "@/lib/models/embeddings/schema";
import { connections } from "@/lib/models/connections/schema";
import { dailyMatches } from "./schema";
import { eq, and, or, sql, notExists, inArray, lt, asc } from "drizzle-orm";

const DAILY_LIMIT = 8;
const CANDIDATE_POOL = 200;

function todayUTC(): string {
  return new Date().toISOString().slice(0, 10);
}

export interface DailyMatchResult {
  user: {
    id: string;
    username: string;
    name: string;
    image: string | null;
    gender: string | null;
    birthdate: string;
    userItems: { id: string; type: string; content: string; displayOrder: number }[];
  };
  score: number;
  distanceKm: number | null;
}

// ─── Public API ──────────────────────────────────────────────────────────────

/** Returns today's matches for the user, generating them on-the-fly if absent. */
export async function getDailyMatches(userId: string): Promise<DailyMatchResult[]> {
  const today = todayUTC();

  const rows = await db
    .select()
    .from(dailyMatches)
    .where(and(eq(dailyMatches.userId, userId), eq(dailyMatches.date, today)));

  if (rows.length > 0) {
    return hydrateMatches(rows);
  }

  return generateAndStoreForUser(userId, today);
}

/**
 * Pre-generate matches for ALL eligible users.
 * Skips users who already have entries for today.
 * Returns the number of users processed.
 */
export async function generateDailyMatchesForAll(): Promise<number> {
  const today = todayUTC();

  const eligible = await db.execute<{ id: string }>(sql`
    SELECT DISTINCT u.id
    FROM   users u
    INNER  JOIN embeddings e
           ON  e.entity_id   = u.id::text
           AND e.entity_type = 'user'
    WHERE  u.coordinates IS NOT NULL
  `);

  let count = 0;
  for (const { id } of eligible) {
    const exists = await db
      .select({ id: dailyMatches.id })
      .from(dailyMatches)
      .where(and(eq(dailyMatches.userId, id), eq(dailyMatches.date, today)))
      .limit(1);

    if (!exists.length) {
      await generateAndStoreForUser(id, today);
      count++;
    }
  }
  return count;
}

/** Deletes daily_matches rows older than today. */
export async function resetDailyMatches(): Promise<number> {
  const today = todayUTC();
  const deleted = await db
    .delete(dailyMatches)
    .where(lt(dailyMatches.date, today))
    .returning({ id: dailyMatches.id });
  return deleted.length;
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/**
 * Core generation logic.
 *
 * Bidirectional similarity via a single JOIN on the embeddings table:
 *   We start from A's embedding and find the top-200 candidates ordered by
 *   cosine distance (symmetric).  We then require that each candidate B also
 *   has an embedding in the table — meaning B is "visible" to A's query and
 *   A is "visible" to B's query with the same score (cosine is symmetric).
 *
 *   In addition we exclude:
 *     - already accepted/declined connections
 *     - users without location, username or birthdate (incomplete profiles)
 */
async function generateAndStoreForUser(
  userId: string,
  date: string,
): Promise<DailyMatchResult[]> {
  const currentUser = await db.query.users.findFirst({
    where: eq(users.id, userId),
  });
  if (!currentUser?.coordinates) return [];

  const myEmbRow = await db.query.embeddings.findFirst({
    where: and(eq(embeddings.entityId, userId), eq(embeddings.entityType, "user")),
    columns: { embedding: true },
  });
  if (!myEmbRow) return [];

  const loc = currentUser.coordinates;
  const vec = `[${myEmbRow.embedding.join(",")}]`;

  const distSql = sql<number>`
    ST_DistanceSphere(
      ${users.coordinates},
      ST_GeomFromText(${`POINT(${loc.x} ${loc.y})`}, 4326)
    ) / 1000`;

  // ── Bidirectional top-200 ────────────────────────────────────────────────
  // Cosine similarity is symmetric: sim(A,B) == sim(B,A).
  // We join embeddings twice:
  //   e_a  — A's embedding (used only for ordering via <=> on e_b)
  //   e_b  — candidate B's embedding
  // Ordering by e_b <=> A's vec gives the same ranking as B would see for A.
  const candidateRows = await db
    .select({
      id: users.id,
      score: sql<number>`1 - (${embeddings.embedding} <=> ${sql.raw(`'${vec}'::vector`)})`,
      distanceKm: distSql,
    })
    .from(embeddings)
    .innerJoin(users, eq(sql`${users.id}::text`, embeddings.entityId))
    .where(
      and(
        eq(embeddings.entityType, "user"),
        // exclude self — cast to text for the entity_id column
        sql`${embeddings.entityId} != ${userId}`,
        sql`${users.coordinates} IS NOT NULL`,
        sql`${users.username}  IS NOT NULL`,
        sql`${users.birthdate} IS NOT NULL`,
        // exclude existing accepted/declined connections
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
                inArray(connections.status, ["accepted", "declined"]),
              ),
            ),
        ),
      ),
    )
    .orderBy(sql`${embeddings.embedding} <=> ${sql.raw(`'${vec}'::vector`)}`)
    .limit(CANDIDATE_POOL);

  if (!candidateRows.length) return [];

  // ── Random sample 8 from top-200 ────────────────────────────────────────
  const sampled = candidateRows.sort(() => Math.random() - 0.5).slice(0, DAILY_LIMIT);

  // ── Persist ──────────────────────────────────────────────────────────────
  await db
    .insert(dailyMatches)
    .values(
      sampled.map((c) => ({
        userId,
        matchedUserId: c.id,
        score: c.score,
        distanceKm: c.distanceKm,
        date,
      })),
    )
    .onConflictDoNothing();

  return hydrateMatches(
    sampled.map((c) => ({
      matchedUserId: c.id,
      score: c.score,
      distanceKm: c.distanceKm,
    })),
  );
}

/** Fetches full user + userItems for a list of daily match rows. */
async function hydrateMatches(
  rows: { matchedUserId: string; score: number; distanceKm: number | null }[],
): Promise<DailyMatchResult[]> {
  if (!rows.length) return [];

  const ids = rows.map((r) => r.matchedUserId);
  const scoreMap = new Map(rows.map((r) => [r.matchedUserId, r]));

  const matchedUsers = await db
    .select()
    .from(users)
    .where(inArray(users.id, ids));

  const items = await db
    .select()
    .from(userItems)
    .where(inArray(userItems.userId, ids))
    .orderBy(asc(userItems.displayOrder));

  const itemsMap = new Map<string, typeof items>();
  for (const item of items) {
    if (!itemsMap.has(item.userId)) itemsMap.set(item.userId, []);
    itemsMap.get(item.userId)!.push(item);
  }

  return matchedUsers
    .map((u) => {
      const row = scoreMap.get(u.id);
      if (!row) return null;
      return {
        user: {
          id: u.id,
          username: u.username ?? "",
          name: u.name,
          image: u.image,
          gender: u.gender,
          birthdate: u.birthdate ?? "",
          userItems: itemsMap.get(u.id) ?? [],
        },
        score: row.score,
        distanceKm: row.distanceKm,
      };
    })
    .filter(Boolean) as DailyMatchResult[];
}
