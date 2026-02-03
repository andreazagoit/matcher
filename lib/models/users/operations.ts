import { db } from "@/lib/db/drizzle";
import { users } from "./schema";
import type { User, NewUser } from "./schema";
import {
  createUserSchema,
  updateUserSchema,
  type CreateUserInput,
  type UpdateUserInput,
} from "./validator";
import { eq } from "drizzle-orm";

/**
 * Operazioni CRUD per Users
 * 
 * NUOVA ARCHITETTURA:
 * - Users: solo dati anagrafici (questa tabella)
 * - Test Sessions/Answers: gestite in tests/operations.ts
 * - Profiles + Matching: gestite in profiles/operations.ts
 * 
 * Gli embeddings e il matching sono stati spostati nel modulo profiles.
 */

/**
 * Crea un nuovo utente (solo dati base)
 */
export async function createUser(input: CreateUserInput): Promise<User> {
  const validatedInput = createUserSchema.parse(input);

  const [newUser] = await db
    .insert(users)
    .values({
      firstName: validatedInput.firstName,
      lastName: validatedInput.lastName,
      email: validatedInput.email,
      birthDate: validatedInput.birthDate,
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
  const validatedInput = updateUserSchema.parse(input);

  const existingUser = await db.query.users.findFirst({
    where: eq(users.id, id),
  });

  if (!existingUser) {
    throw new Error("User not found");
  }

  const updateData: Partial<NewUser> = {
    updatedAt: new Date(),
  };

  if (validatedInput.firstName !== undefined) {
    updateData.firstName = validatedInput.firstName;
  }
  if (validatedInput.lastName !== undefined) {
    updateData.lastName = validatedInput.lastName;
  }
  if (validatedInput.email !== undefined) {
    updateData.email = validatedInput.email;
  }
  if (validatedInput.birthDate !== undefined) {
    updateData.birthDate = validatedInput.birthDate;
  }

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
