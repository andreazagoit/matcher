import { z } from "zod";

/**
 * Zod validation schemas for user-related data.
 */

const usernameSchema = z
  .string()
  .min(3, "Username must be at least 3 characters")
  .max(30, "Username must be at most 30 characters")
  .regex(/^[a-z0-9_]+$/, "Username can only contain lowercase letters, numbers and underscores");

// Schema for creating a user (base data only, OIDC naming)
export const createUserSchema = z.object({
  username: usernameSchema,
  givenName: z.string().min(1, "Given name is required"),
  familyName: z.string().min(1, "Family name is required"),
  email: z.string().email("Invalid email"),
  birthdate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, "Date must be YYYY-MM-DD"),
  gender: z.enum(["man", "woman", "non_binary"]).optional(),
});

// Schema for the sign-up form — gender is required
export const signUpSchema = createUserSchema.extend({
  gender: z.enum(["man", "woman", "non_binary"], { message: "Seleziona il genere" }),
});

// Schema for updating a user (all fields optional)
export const updateUserSchema = z.object({
  username: usernameSchema.optional(),
  givenName: z.string().min(1).optional(),
  familyName: z.string().min(1).optional(),
  email: z.string().email().optional(),
  birthdate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).optional(),
  gender: z.enum(["man", "woman", "non_binary"]).optional(),
});

export type CreateUserInput = z.infer<typeof createUserSchema>;
export type UpdateUserInput = z.infer<typeof updateUserSchema>;

