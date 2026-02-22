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
import { GraphQLError } from "graphql";

/**
 * Create a new user record.
 */
export async function createUser(
  input: CreateUserInput
): Promise<User> {
  const validatedInput = createUserSchema.parse(input);

  const existing = await db.query.users.findFirst({
    where: eq(users.username, validatedInput.username),
  });
  if (existing) {
    throw new GraphQLError("Username already taken", { extensions: { code: "USERNAME_TAKEN" } });
  }

  const [newUser] = await db
    .insert(users)
    .values({
      username: validatedInput.username,
      name: validatedInput.name,
      email: validatedInput.email,
      birthdate: validatedInput.birthdate,
      ...(validatedInput.gender && { gender: validatedInput.gender }),
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

  if (validatedInput.username !== undefined) {
    const taken = await db.query.users.findFirst({ where: eq(users.username, validatedInput.username) });
    if (taken && taken.id !== id) {
      throw new GraphQLError("Username already taken", { extensions: { code: "USERNAME_TAKEN" } });
    }
    updateData.username = validatedInput.username;
  }
  if (validatedInput.name !== undefined) updateData.name = validatedInput.name;
  if (validatedInput.email !== undefined) updateData.email = validatedInput.email;
  if (validatedInput.birthdate !== undefined) updateData.birthdate = validatedInput.birthdate;
  if (validatedInput.gender !== undefined) updateData.gender = validatedInput.gender;

  // Orientation & identity
  if (validatedInput.sexualOrientation !== undefined) updateData.sexualOrientation = validatedInput.sexualOrientation;
  if (validatedInput.heightCm !== undefined) updateData.heightCm = validatedInput.heightCm;

  // Relational intent
  if (validatedInput.relationshipIntent !== undefined) updateData.relationshipIntent = validatedInput.relationshipIntent;
  if (validatedInput.relationshipStyle !== undefined) updateData.relationshipStyle = validatedInput.relationshipStyle;
  if (validatedInput.hasChildren !== undefined) updateData.hasChildren = validatedInput.hasChildren;
  if (validatedInput.wantsChildren !== undefined) updateData.wantsChildren = validatedInput.wantsChildren;

  // Lifestyle
  if (validatedInput.religion !== undefined) updateData.religion = validatedInput.religion;
  if (validatedInput.smoking !== undefined) updateData.smoking = validatedInput.smoking;
  if (validatedInput.drinking !== undefined) updateData.drinking = validatedInput.drinking;
  if (validatedInput.activityLevel !== undefined) updateData.activityLevel = validatedInput.activityLevel;
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
 * Retrieve a user by their username.
 */
export async function getUserByUsername(username: string): Promise<User | null> {
  const result = await db.query.users.findFirst({
    where: eq(users.username, username),
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
 * Check if a username is already taken. No auth required (used at sign-up).
 */
export async function isUsernameTaken(username: string): Promise<boolean> {
  const existing = await db.query.users.findFirst({
    where: eq(users.username, username),
    columns: { id: true },
  });
  return !!existing;
}

/**
 * Update a user's location (PostGIS point).
 * PostGIS convention: x = longitude, y = latitude.
 */
export async function updateUserLocation(
  id: string,
  lat: number,
  lon: number,
): Promise<User> {
  const [updated] = await db
    .update(users)
    .set({
      location: { x: lon, y: lat },
      locationUpdatedAt: new Date(),
      updatedAt: new Date(),
    })
    .where(eq(users.id, id))
    .returning();

  if (!updated) throw new Error("User not found");
  return updated;
}

