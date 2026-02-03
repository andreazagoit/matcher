import { db } from "@/lib/db/drizzle";
import { users } from "./schema";
import type { User, NewUser } from "./schema";
import type { Value } from "@/lib/models/values/operations";
import type { Interest } from "@/lib/models/interests/operations";
import { generateUserEmbeddings } from "@/lib/embeddings";
import {
  createUserSchema,
  updateUserSchema,
  type CreateUserInput,
  type UpdateUserInput,
} from "./validator";
import { eq, sql, desc, ne, cosineDistance } from "drizzle-orm";

/**
 * Crea un nuovo utente con embeddings generati automaticamente
 */
export async function createUser(
  input: CreateUserInput
): Promise<User> {
  // Valida input con Zod
  const validatedInput = createUserSchema.parse(input);

  // Genera embeddings
  const { valuesEmbedding, interestsEmbedding } =
    await generateUserEmbeddings(validatedInput.values, validatedInput.interests);

  // Crea utente con embeddings
  const [newUser] = await db
    .insert(users)
    .values({
      firstName: validatedInput.firstName,
      lastName: validatedInput.lastName,
      email: validatedInput.email,
      birthDate: validatedInput.birthDate,
      values: validatedInput.values as Value[],
      interests: validatedInput.interests as Interest[],
      valuesEmbedding,
      interestsEmbedding,
    })
    .returning();

  return newUser;
}

/**
 * Aggiorna un utente esistente
 */
export async function updateUser(
  id: string,
  input: UpdateUserInput
): Promise<User> {
  // Valida input con Zod
  const validatedInput = updateUserSchema.parse(input);

  const existingUser = await db.query.users.findFirst({
    where: eq(users.id, id),
  });

  if (!existingUser) {
    throw new Error("User not found");
  }

  // Se values o interests vengono aggiornati, rigenera embeddings
  const updatedValues = validatedInput.values ?? existingUser.values;
  const updatedInterests = validatedInput.interests ?? existingUser.interests;

  let valuesEmbedding = existingUser.valuesEmbedding;
  let interestsEmbedding = existingUser.interestsEmbedding;

  if (validatedInput.values || validatedInput.interests) {
    const embeddings = await generateUserEmbeddings(
      updatedValues,
      updatedInterests
    );
    valuesEmbedding = embeddings.valuesEmbedding;
    interestsEmbedding = embeddings.interestsEmbedding;
  }

  const updateData: Partial<NewUser> = {
    updatedAt: new Date(),
    valuesEmbedding,
    interestsEmbedding,
  };

  if (validatedInput.firstName !== undefined) updateData.firstName = validatedInput.firstName;
  if (validatedInput.lastName !== undefined) updateData.lastName = validatedInput.lastName;
  if (validatedInput.email !== undefined) updateData.email = validatedInput.email;
  if (validatedInput.birthDate !== undefined) updateData.birthDate = validatedInput.birthDate;
  if (validatedInput.values !== undefined) updateData.values = validatedInput.values as Value[];
  if (validatedInput.interests !== undefined) updateData.interests = validatedInput.interests as Interest[];

  const [updatedUser] = await db
    .update(users)
    .set(updateData)
    .where(eq(users.id, id))
    .returning();

  return updatedUser;
}

/**
 * Trova utente per ID
 */
export async function getUserById(id: string): Promise<User | null> {
  const result = await db.query.users.findFirst({
    where: eq(users.id, id),
  });
  return result || null;
}

/**
 * Trova utente per email
 */
export async function getUserByEmail(email: string): Promise<User | null> {
  const result = await db.query.users.findFirst({
    where: eq(users.email, email),
  });
  return result || null;
}

/**
 * Ottieni tutti gli utenti
 */
export async function getAllUsers(): Promise<User[]> {
  return await db.query.users.findMany();
}

/**
 * Elimina un utente
 */
export async function deleteUser(id: string): Promise<boolean> {
  const [deleted] = await db
    .delete(users)
    .where(eq(users.id, id))
    .returning();
  return !!deleted;
}

export interface UserMatch extends User {
  similarity: number;
}

export interface FindMatchesOptions {
  limit?: number;
  /** Numero di candidati da recuperare con ANN prima del re-ranking */
  candidates?: number;
}

/**
 * Trova match per un utente usando ANN search con pgvector (2-stage retrieval)
 * 
 * ⚠️ REGOLA FONDAMENTALE:
 * ANN serve SOLO a scegliere i candidati (usa indice HNSW).
 * Il ranking con i pesi si fa DOPO, in memoria.
 * 
 * - ANN può usare UN SOLO indice vettoriale per query
 * - ORDER BY su combinazione di cosine → full scan (NO!)
 * - I pesi sono logica di business, non di indice
 * 
 * Pipeline:
 * 1. ANN su embedding dominante → candidati (~200)
 * 2. Calcolo similarità su tutti gli embedding (solo sui candidati)
 * 3. Ranking finale pesato in memoria
 */
export async function findMatches(
  userId: string,
  options: FindMatchesOptions = {}
): Promise<UserMatch[]> {
  // Pesi per il ranking finale (logica di business)
  // L'embedding con peso maggiore guida l'ANN
  const VALUES_WEIGHT = 0.7;
  const INTERESTS_WEIGHT = 0.3;

  const { limit = 10, candidates: CANDIDATES = 200 } = options;

  const currentUser = await getUserById(userId);
  if (!currentUser) {
    throw new Error("User not found");
  }

  const valuesEmbedding = currentUser.valuesEmbedding as number[];
  const interestsEmbedding = currentUser.interestsEmbedding as number[];

  // ========================================
  // STAGE 1: ANN Search (usa indice HNSW)
  // ========================================
  // 🔑 ORDER BY diretto su cosineDistance → Postgres usa HNSW
  // NON scansiona tutta la tabella, trova solo utenti "vicini"
  // Verifica con: EXPLAIN ANALYZE → deve mostrare "Index Scan using ... hnsw"
  const candidates = await db
    .select({
      id: users.id,
      firstName: users.firstName,
      lastName: users.lastName,
      email: users.email,
      birthDate: users.birthDate,
      values: users.values,
      interests: users.interests,
      valuesEmbedding: users.valuesEmbedding,
      interestsEmbedding: users.interestsEmbedding,
      createdAt: users.createdAt,
      updatedAt: users.updatedAt,
      // Calcola similarità per entrambi gli embedding (solo sui ~200 candidati)
      valuesSimilarity: sql<number>`1 - (${cosineDistance(users.valuesEmbedding, valuesEmbedding)})`,
      interestsSimilarity: sql<number>`1 - (${cosineDistance(users.interestsEmbedding, interestsEmbedding)})`,
    })
    .from(users)
    .where(ne(users.id, userId))
    // 🔑 QUESTO USA ANN (HNSW) - ordina per embedding dominante
    .orderBy(sql`${cosineDistance(users.valuesEmbedding, valuesEmbedding)}`)
    .limit(CANDIDATES);

  // ========================================
  // STAGE 2: Ranking finale (logica di business)
  // ========================================
  // Applica i pesi SOLO sui candidati (~200) - velocissimo in memoria
  const matches = candidates
    .map((u) => ({
      ...u,
      similarity:
        VALUES_WEIGHT * u.valuesSimilarity +
        INTERESTS_WEIGHT * u.interestsSimilarity,
    }))
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, limit);

  return matches as UserMatch[];
}

/**
 * Trova match usando MULTI-ANN (qualità migliore)
 * 
 * Esegue ANN su ENTRAMBI gli embedding, unisce i candidati,
 * poi applica il ranking pesato finale.
 * 
 * Pro: migliore recall (trova match che sarebbero persi con single-ANN)
 * Con: 2 query invece di 1 (ma comunque velocissimo)
 */
export async function findMatchesMultiANN(
  userId: string,
  options: FindMatchesOptions = {}
): Promise<UserMatch[]> {
  const VALUES_WEIGHT = 0.7;
  const INTERESTS_WEIGHT = 0.3;

  const { limit = 10, candidates: CANDIDATES_PER_INDEX = 150 } = options;

  const currentUser = await getUserById(userId);
  if (!currentUser) {
    throw new Error("User not found");
  }

  const valuesEmbedding = currentUser.valuesEmbedding as number[];
  const interestsEmbedding = currentUser.interestsEmbedding as number[];

  // ========================================
  // STAGE 1: Multi-ANN Search (2 query parallele)
  // ========================================
  // Esegue ANN su entrambi gli indici HNSW in parallelo
  const [candidatesValues, candidatesInterests] = await Promise.all([
    // ANN su values embedding
    db
      .select({
        id: users.id,
        firstName: users.firstName,
        lastName: users.lastName,
        email: users.email,
        birthDate: users.birthDate,
        values: users.values,
        interests: users.interests,
        valuesEmbedding: users.valuesEmbedding,
        interestsEmbedding: users.interestsEmbedding,
        createdAt: users.createdAt,
        updatedAt: users.updatedAt,
      })
      .from(users)
      .where(ne(users.id, userId))
      .orderBy(sql`${cosineDistance(users.valuesEmbedding, valuesEmbedding)}`)
      .limit(CANDIDATES_PER_INDEX),

    // ANN su interests embedding
    db
      .select({
        id: users.id,
        firstName: users.firstName,
        lastName: users.lastName,
        email: users.email,
        birthDate: users.birthDate,
        values: users.values,
        interests: users.interests,
        valuesEmbedding: users.valuesEmbedding,
        interestsEmbedding: users.interestsEmbedding,
        createdAt: users.createdAt,
        updatedAt: users.updatedAt,
      })
      .from(users)
      .where(ne(users.id, userId))
      .orderBy(sql`${cosineDistance(users.interestsEmbedding, interestsEmbedding)}`)
      .limit(CANDIDATES_PER_INDEX),
  ]);

  // ========================================
  // STAGE 2: Merge candidati (union by id)
  // ========================================
  const candidatesMap = new Map<string, typeof candidatesValues[number]>();
  
  for (const c of candidatesValues) {
    candidatesMap.set(c.id, c);
  }
  for (const c of candidatesInterests) {
    if (!candidatesMap.has(c.id)) {
      candidatesMap.set(c.id, c);
    }
  }

  const mergedCandidates = Array.from(candidatesMap.values());

  // ========================================
  // STAGE 3: Calcolo similarità + Ranking finale
  // ========================================
  const matches = mergedCandidates
    .map((u) => {
      // Calcola cosine similarity in memoria
      const valuesSim = cosineSimilarity(
        u.valuesEmbedding as number[],
        valuesEmbedding
      );
      const interestsSim = cosineSimilarity(
        u.interestsEmbedding as number[],
        interestsEmbedding
      );

      return {
        ...u,
        similarity: VALUES_WEIGHT * valuesSim + INTERESTS_WEIGHT * interestsSim,
      };
    })
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, limit);

  return matches as UserMatch[];
}

/**
 * Calcola cosine similarity tra due vettori
 * similarity = dot(a,b) / (||a|| * ||b||)
 * Range: [-1, 1] dove 1 = identici
 */
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  if (denominator === 0) return 0;

  return dot / denominator;
}

