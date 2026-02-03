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
}

/**
 * Trova match per un utente usando ANN search con pgvector
 */
export async function findMatches(
  userId: string,
  options: FindMatchesOptions = {}
): Promise<UserMatch[]> {
  // Pesi per il calcolo della similarità nei match
  const VALUES_WEIGHT = 0.5;
  const INTERESTS_WEIGHT = 0.5;

  const { limit = 10 } = options;

  // Trova l'utente corrente
  const currentUser = await getUserById(userId);

  if (!currentUser) {
    throw new Error("User not found");
  }

  // Calcola similarità usando cosineDistance di Drizzle
  // Seguendo il pattern della documentazione ufficiale
  const valuesEmbedding = currentUser.valuesEmbedding as number[];
  const interestsEmbedding = currentUser.interestsEmbedding as number[];
  
  // Calcola le due similarità separatamente (1 - distanza = similarità)
  const valuesSimilarity = sql<number>`1 - (${cosineDistance(users.valuesEmbedding, valuesEmbedding)})`;
  const interestsSimilarity = sql<number>`1 - (${cosineDistance(users.interestsEmbedding, interestsEmbedding)})`;
  
  // Similarità combinata pesata
  const similarity = sql<number>`(${valuesSimilarity} * ${VALUES_WEIGHT} + ${interestsSimilarity} * ${INTERESTS_WEIGHT})`;

  const matches = await db
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
      similarity,
    })
    .from(users)
    .where(ne(users.id, userId))
    .orderBy((t) => desc(t.similarity))
    .limit(limit);

  return matches as UserMatch[];
}

