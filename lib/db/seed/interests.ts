import { db } from "../drizzle";
import { userInterests } from "../../models/interests/schema";
import { embeddings } from "../../models/embeddings/schema";
import { users } from "../../models/users/schema";
import { ALL_TAGS } from "../../models/tags/data";
import { generateEmbeddingsBatch } from "../../embeddings";

function pickRandom<T>(arr: T[], count: number): T[] {
  return [...arr].sort(() => Math.random() - 0.5).slice(0, count);
}

function buildUserText(tags: string[], birthdate?: string | null): string {
  const parts: string[] = [];
  if (tags.length > 0) parts.push(`Interests: ${tags.join(", ")}`);
  if (birthdate) {
    const age = new Date().getFullYear() - new Date(birthdate).getFullYear();
    parts.push(`Age: ${age}`);
  }
  return parts.join(". ");
}

/**
 * Seed interests and behavioral embeddings for a list of user IDs.
 * Each user gets 5–10 random interests and an embedding stored in the
 * shared `embeddings` table, enabling AI-powered recommendations after db:reset.
 */
export async function seedProfiles(userIds: string[]) {
  console.log(`\n📋 Seeding ${userIds.length} users' interests + embeddings...`);

  // 1. Assign random interests to each user
  const userTagMap: Record<string, string[]> = {};

  for (const userId of userIds) {
    const tags = pickRandom(ALL_TAGS, 5 + Math.floor(Math.random() * 6));
    for (const tag of tags) {
      await db.insert(userInterests).values({ userId, tag, weight: 1.0 });
    }
    userTagMap[userId] = tags;
  }

  console.log(`  → ${userIds.length} users' interests created`);

  // 2. Fetch birthdates in one query
  const userRows = await db.query.users.findMany({
    where: (u, { inArray }) => inArray(u.id, userIds),
    columns: { id: true, birthdate: true },
  });
  const birthdateMap = Object.fromEntries(userRows.map((u) => [u.id, u.birthdate]));

  // 3. Build embedding texts and batch-generate via single OpenAI request
  const texts = userIds.map((userId) =>
    buildUserText(userTagMap[userId], birthdateMap[userId])
  );

  console.log(`  → Generating embeddings (batch)...`);

  try {
    const vectors = await generateEmbeddingsBatch(texts);

    // 4. Upsert all embeddings into the shared embeddings table
    for (let i = 0; i < userIds.length; i++) {
      await db
        .insert(embeddings)
        .values({ entityId: userIds[i], entityType: "user", embedding: vectors[i] })
        .onConflictDoUpdate({
          target: [embeddings.entityId, embeddings.entityType],
          set: { embedding: vectors[i], updatedAt: new Date() },
        });
    }

    console.log(`  → ${userIds.length} behavioral embeddings stored`);
  } catch (err) {
    console.warn(`  ⚠️  Batch embedding failed, interests created without embeddings:`, err);
  }
}
