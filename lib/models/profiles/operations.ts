import { db } from "@/lib/db/drizzle";
import { eq, ne, sql, and, inArray } from "drizzle-orm";
import { cosineDistance } from "drizzle-orm";
import { users } from "@/lib/models/users/schema";
import {
  profiles,
  DEFAULT_MATCHING_WEIGHTS,
  type Profile,
} from "./schema";
import { generateAllUserEmbeddings } from "@/lib/embeddings";

// ============================================
// PROFILE CRUD
// ============================================

export interface ProfileData {
  psychologicalDesc: string;
  valuesDesc: string;
  interestsDesc: string;
  behavioralDesc: string;
}

/**
 * Create or update a user profile.
 * Automatically generates vector embeddings from the provided descriptions.
 */
export async function upsertProfile(
  userId: string,
  data: ProfileData,
  assessmentVersion: number = 1
): Promise<Profile> {
  // Generate vector embeddings from textual descriptions via OpenAI
  const embeddings = await generateAllUserEmbeddings({
    psychological: data.psychologicalDesc,
    values: data.valuesDesc,
    interests: data.interestsDesc,
    behavioral: data.behavioralDesc,
  });

  const now = new Date();

  // Execute Profile Upsert (INSERT or UPDATE on unique constraint conflict)
  const [profile] = await db
    .insert(profiles)
    .values({
      userId,
      psychologicalDesc: data.psychologicalDesc,
      valuesDesc: data.valuesDesc,
      interestsDesc: data.interestsDesc,
      behavioralDesc: data.behavioralDesc,
      psychologicalEmbedding: embeddings.psychological,
      valuesEmbedding: embeddings.values,
      interestsEmbedding: embeddings.interests,
      behavioralEmbedding: embeddings.behavioral,
      assessmentVersion,
      updatedAt: now,
    })
    .onConflictDoUpdate({
      target: profiles.userId,
      set: {
        psychologicalDesc: data.psychologicalDesc,
        valuesDesc: data.valuesDesc,
        interestsDesc: data.interestsDesc,
        behavioralDesc: data.behavioralDesc,
        psychologicalEmbedding: embeddings.psychological,
        valuesEmbedding: embeddings.values,
        interestsEmbedding: embeddings.interests,
        behavioralEmbedding: embeddings.behavioral,
        assessmentVersion,
        updatedAt: now,
      },
    })
    .returning();

  return profile;
}

/**
 * Retrieve a user profile by their User ID.
 */
export async function getProfileByUserId(
  userId: string
): Promise<Profile | null> {
  const result = await db.query.profiles.findFirst({
    where: eq(profiles.userId, userId),
  });
  return result || null;
}

/**
 * Verify if a user has a fully generated and complete profile.
 */
export async function hasCompleteProfile(userId: string): Promise<boolean> {
  const profile = await getProfileByUserId(userId);
  return !!(profile?.psychologicalEmbedding);
}

// ============================================
// MATCHING
// ============================================

export interface ProfileMatch {
  profile: Profile;
  user: typeof users.$inferSelect;
  similarity: number;
  breakdown: {
    psychological: number;
    values: number;
    interests: number;
    behavioral: number;
  };
}

export interface FindMatchesOptions {
  limit?: number;
  weights?: Record<keyof typeof DEFAULT_MATCHING_WEIGHTS, number>;
  /** Gender filter ("man", "woman", "non_binary") */
  gender?: ("man" | "woman" | "non_binary")[];
  /** Minimum age filter (inclusive) */
  minAge?: number;
  /** Maximum age filter (inclusive) */
  maxAge?: number;
}

/**
 * Find compatible matches using a combination of ANN search and weighted ranking.
 * 
 * Matching Logic:
 * 1. ANN Search on 'psychological' axis (dominant) to narrow down to top candidates.
 * 2. Calculate fine-grained similarity across all axes.
 * 3. Final weighted ranking based on configured matching weights.
 */
export async function findMatches(
  userId: string,
  options: FindMatchesOptions = {}
): Promise<ProfileMatch[]> {
  const {
    limit = 10,
    weights = DEFAULT_MATCHING_WEIGHTS,
    gender,
    minAge,
    maxAge,
  } = options;

  /** Initial candidate pool size from ANN (Stage 1) */
  const CANDIDATES = 200;

  const currentProfile = await getProfileByUserId(userId);
  if (!currentProfile?.psychologicalEmbedding) {
    throw new Error("Profile not found. Complete the assessment first.");
  }

  const psychEmbedding = currentProfile.psychologicalEmbedding as number[];
  const valuesEmbedding = currentProfile.valuesEmbedding as number[] | null;
  const interestsEmbedding = currentProfile.interestsEmbedding as number[] | null;
  const behavioralEmbedding = currentProfile.behavioralEmbedding as number[] | null;

  // Build dynamic SQL where conditions
  const filterConditions = [ne(profiles.userId, userId)];

  // Apply gender filtering if specified
  if (gender && gender.length > 0) {
    // Filtra solo utenti con gender non null e che corrisponde ai valori richiesti
    filterConditions.push(inArray(users.gender, gender));
  }

  // Age calculated from birthdate (using SQL interval logic)
  if (minAge !== undefined) {
    filterConditions.push(
      sql`EXTRACT(YEAR FROM AGE(${users.birthdate})) >= ${minAge}`
    );
  }

  if (maxAge !== undefined) {
    filterConditions.push(
      sql`EXTRACT(YEAR FROM AGE(${users.birthdate})) <= ${maxAge}`
    );
  }

  // STAGE 1: Efficient ANN Search using HNSW indices
  const whereClause = filterConditions.length > 1
    ? and(...filterConditions)!
    : filterConditions[0];

  const candidateProfiles = await db
    .select({
      profile: profiles,
      user: users,
      psychSimilarity: sql<number>`1 - (${cosineDistance(
        profiles.psychologicalEmbedding,
        psychEmbedding
      )})`,
    })
    .from(profiles)
    .innerJoin(users, eq(profiles.userId, users.id))
    .where(whereClause)
    .orderBy(sql`${cosineDistance(profiles.psychologicalEmbedding, psychEmbedding)}`)
    .limit(CANDIDATES);

  // STAGE 2: Multi-axis weighted ranking
  const matches: ProfileMatch[] = candidateProfiles.map((candidate) => {
    const cp = candidate.profile;
    const psychSim = candidate.psychSimilarity;

    const valuesSim = valuesEmbedding && cp.valuesEmbedding
      ? cosineSimilarity(valuesEmbedding, cp.valuesEmbedding as number[])
      : 0;

    const interestsSim = interestsEmbedding && cp.interestsEmbedding
      ? cosineSimilarity(interestsEmbedding, cp.interestsEmbedding as number[])
      : 0;

    const behavioralSim = behavioralEmbedding && cp.behavioralEmbedding
      ? cosineSimilarity(behavioralEmbedding, cp.behavioralEmbedding as number[])
      : 0;

    const similarity =
      weights.psychological * psychSim +
      weights.values * valuesSim +
      weights.interests * interestsSim +
      weights.behavioral * behavioralSim;

    return {
      profile: cp,
      user: candidate.user,
      similarity,
      breakdown: {
        psychological: psychSim,
        values: valuesSim,
        interests: interestsSim,
        behavioral: behavioralSim,
      },
    };
  });

  return matches
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, limit);
}

// ============================================
// UTILITY
// ============================================

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}
