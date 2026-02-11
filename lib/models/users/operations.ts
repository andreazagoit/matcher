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
 * Create a new user record.
 */
export async function createUser(
  input: CreateUserInput
): Promise<User> {
  // Validate input with Zod
  const validatedInput = createUserSchema.parse(input);

  // Create user
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
 * Update an existing user record with partial data.
 */
export async function updateUser(
  id: string,
  input: UpdateUserInput
): Promise<User> {
  // Validate input with Zod
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

  if (validatedInput.firstName !== undefined) updateData.firstName = validatedInput.firstName;
  if (validatedInput.lastName !== undefined) updateData.lastName = validatedInput.lastName;
  if (validatedInput.email !== undefined) updateData.email = validatedInput.email;
  if (validatedInput.birthDate !== undefined) updateData.birthDate = validatedInput.birthDate;

  const [updatedUser] = await db
    .update(users)
    .set(updateData)
    .where(eq(users.id, id))
    .returning();

  return updatedUser;
}

/**
 * Retrieve a user by their unique ID.
 */
export async function getUserById(id: string): Promise<User | null> {
  const result = await db.query.users.findFirst({
    where: eq(users.id, id),
  });
  return result || null;
}

/**
 * Retrieve a user by their email address.
 */
export async function getUserByEmail(email: string): Promise<User | null> {
  const result = await db.query.users.findFirst({
    where: eq(users.email, email),
  });
  return result || null;
}

/**
 * Retrieve all user records from the database.
 */
export async function getAllUsers(): Promise<User[]> {
  return await db.query.users.findMany();
}

/**
 * Permanently delete a user record by ID.
 */
export async function deleteUser(id: string): Promise<boolean> {
  const [deleted] = await db
    .delete(users)
    .where(eq(users.id, id))
    .returning();
  return !!deleted;
}

/**
 * @deprecated Use profiles system for matching.
 */

