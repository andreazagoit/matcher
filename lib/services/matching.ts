import { db } from "@/lib/db/drizzle";
import { profiles } from "@/lib/models/profiles/schema";
import { users } from "@/lib/models/users/schema";
import { connections } from "@/lib/models/connections/schema";
import { dailyMatches } from "@/lib/models/matches/schema";
import { eq, notInArray, sql, desc, or } from "drizzle-orm";
import { cosineDistance } from "drizzle-orm/sql";

export async function getDailyMatches(userId: string) {
  // 1. Check if matches already exist
  const existingMatches = await db.query.dailyMatches.findMany({
    where: eq(dailyMatches.userId, userId),
    with: {
      match: {
        with: {
          profile: true
        }
      }
    }
  });

  if (existingMatches.length > 0) {
    return existingMatches.map(m => m.match);
  }

  // 2. Generate new matches
  return generateDailyMatches(userId);
}

async function generateDailyMatches(userId: string) {
  // Fetch existing user profile and its pre-computed embeddings
  const userProfile = await db.query.profiles.findFirst({
    where: eq(profiles.userId, userId),
  });

  if (!userProfile || !userProfile.psychologicalEmbedding) {
    return []; // Profile or embeddings are incomplete
  }

  // Exclude users with existing connections or rejections to avoid repetition
  const existingConnections = await db.query.connections.findMany({
    where: or(
      eq(connections.requesterId, userId),
      eq(connections.targetId, userId)
    ),
  });

  const excludedIds = new Set<string>();
  excludedIds.add(userId); // Exclude self from matches

  existingConnections.forEach(c => {
    excludedIds.add(c.requesterId === userId ? c.targetId : c.requesterId);
  });

  // Find compatible candidates using the Psychological Embedding (dominant match factor)
  // Strategy: 
  // 1. Vector Search for top candidates by similarity
  // 2. Filter out already connected/excluded users
  // 3. Selection of top matches for final processing

  const similarity = sql<number>`1 - (${cosineDistance(
    profiles.psychologicalEmbedding,
    userProfile.psychologicalEmbedding
  )})`;

  const candidates = await db
    .select({
      id: users.id,
      firstName: users.firstName,
      lastName: users.lastName,
      image: users.image,
      compatibility: similarity,
    })
    .from(users)
    .innerJoin(profiles, eq(users.id, profiles.userId))
    .where(
      notInArray(users.id, Array.from(excludedIds))
    )
    .orderBy(desc(similarity))
    .limit(50); // Top 50 compatible

  // Shuffle candidates and select a subset (e.g., top 3) for the daily set
  const shuffled = candidates.sort(() => 0.5 - Math.random());
  const selected = shuffled.slice(0, 3);

  // Save to DB
  if (selected.length > 0) {
    await db.insert(dailyMatches).values(
      selected.map(match => ({
        userId,
        matchId: match.id,
      }))
    );
  }

  // Return the selected matches in a format consistent with User objects
  return selected;
}
