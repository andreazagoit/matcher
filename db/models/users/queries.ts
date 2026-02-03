import { db } from "../../index";
import { users } from "./schema";
import type { User, NewUser } from "./schema";
import type { Value } from "../values/data";
import type { Interest } from "../interests/data";
import { generateUserEmbeddings } from "@/lib/embeddings";
import {
  createUserSchema,
  updateUserSchema,
  type CreateUserInput,
  type UpdateUserInput,
} from "./validator";
import { eq, sql } from "drizzle-orm";

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
  valuesWeight?: number;
  interestsWeight?: number;
}

/**
 * Trova match per un utente usando ANN search con pgvector
 */
export async function findMatches(
  userId: string,
  options: FindMatchesOptions = {}
): Promise<UserMatch[]> {
  const {
    limit = 10,
    valuesWeight = 0.5,
    interestsWeight = 0.5,
  } = options;

  // Trova l'utente corrente
  const currentUser = await getUserById(userId);

  if (!currentUser) {
    throw new Error("User not found");
  }

  // ANN search usando cosine distance
  const valuesEmbeddingJson = JSON.stringify(currentUser.valuesEmbedding);
  const interestsEmbeddingJson = JSON.stringify(currentUser.interestsEmbedding);

  const matches = await db.execute(sql`
    SELECT 
      id,
      first_name AS "firstName",
      last_name AS "lastName",
      email,
      birth_date AS "birthDate",
      values,
      interests,
      values_embedding AS "valuesEmbedding",
      interests_embedding AS "interestsEmbedding",
      created_at AS "createdAt",
      updated_at AS "updatedAt",
      (
        (1 - (values_embedding <=> ${valuesEmbeddingJson}::vector)) * ${valuesWeight} +
        (1 - (interests_embedding <=> ${interestsEmbeddingJson}::vector)) * ${interestsWeight}
      ) AS similarity
    FROM users
    WHERE id != ${userId}
    ORDER BY similarity DESC
    LIMIT ${limit}
  `);

  return matches as unknown as UserMatch[];
}

