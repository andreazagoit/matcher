import { db } from "@/lib/db/drizzle";
import { eq, ne, sql, and, inArray } from "drizzle-orm";
import { cosineDistance } from "drizzle-orm";
import { users } from "@/lib/models/users/schema";
import {
  profiles,
  DEFAULT_MATCHING_WEIGHTS,
  type Profile,
  type ProfileAxis,
} from "./schema";
import { generateAllUserEmbeddings } from "@/lib/embeddings";

// ============================================
// PROFILE CRUD
// ============================================

export interface ProfileData {
  psychological: ProfileAxis;
  values: ProfileAxis;
  interests: ProfileAxis;
  behavioral: ProfileAxis;
}

/**
 * Crea o aggiorna il profilo utente
 * Genera automaticamente gli embeddings dalle descrizioni
 */
export async function upsertProfile(
  userId: string,
  data: ProfileData,
  assessmentVersion: number = 1
): Promise<Profile> {
  // Genera embeddings da descrizioni testuali
  const embeddings = await generateAllUserEmbeddings({
    psychological: data.psychological.description,
    values: data.values.description,
    interests: data.interests.description,
    behavioral: data.behavioral.description,
  });

  const now = new Date();

  // Upsert profilo (INSERT or UPDATE on conflict)
  const [profile] = await db
    .insert(profiles)
    .values({
      userId,
      psychological: data.psychological,
      values: data.values,
      interests: data.interests,
      behavioral: data.behavioral,
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
        psychological: data.psychological,
        values: data.values,
        interests: data.interests,
        behavioral: data.behavioral,
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
 * Ottieni profilo per userId
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
 * Verifica se un utente ha un profilo completo
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
  /** Filtro per genere (array di valori: "man", "woman", "non_binary") */
  gender?: ("man" | "woman" | "non_binary")[];
  /** Età minima (inclusiva) */
  minAge?: number;
  /** Età massima (inclusiva) */
  maxAge?: number;
}

/**
 * Trova match usando ANN Search + Ranking Pesato
 * 
 * Pipeline:
 * 1. ANN su psychological (dominante) → 200 candidati
 * 2. Calcolo similarità per tutti gli assi
 * 3. Ranking finale pesato
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

  /** Numero fisso di candidati da ANN (stage 1) */
  const CANDIDATES = 200;

  const currentProfile = await getProfileByUserId(userId);
  if (!currentProfile?.psychologicalEmbedding) {
    throw new Error("Profile not found. Complete the assessment first.");
  }

  const psychEmbedding = currentProfile.psychologicalEmbedding as number[];
  const valuesEmbedding = currentProfile.valuesEmbedding as number[] | null;
  const interestsEmbedding = currentProfile.interestsEmbedding as number[] | null;
  const behavioralEmbedding = currentProfile.behavioralEmbedding as number[] | null;

  // Costruisci condizioni filtro
  const filterConditions = [ne(profiles.userId, userId)];

  // Filtro genere (solo se specificato)
  if (gender && gender.length > 0) {
    // Filtra solo utenti con gender non null e che corrisponde ai valori richiesti
    filterConditions.push(inArray(users.gender, gender));
  }

  // Filtro età (calcola età da birthDate)
  if (minAge !== undefined) {
    filterConditions.push(
      sql`EXTRACT(YEAR FROM AGE(${users.birthDate})) >= ${minAge}`
    );
  }

  if (maxAge !== undefined) {
    filterConditions.push(
      sql`EXTRACT(YEAR FROM AGE(${users.birthDate})) <= ${maxAge}`
    );
  }

  // STAGE 1: ANN Search (usa HNSW) con filtri
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

  // STAGE 2: Ranking pesato
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
