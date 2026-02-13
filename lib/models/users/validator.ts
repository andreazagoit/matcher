import { z } from "zod";

/**
 * Zod validation schemas for user-related data.
 */

// Schema for creating a user (base data only)
export const createUserSchema = z.object({
  firstName: z.string().min(1, "First name is required"),
  lastName: z.string().min(1, "Last name is required"),
  email: z.string().email("Invalid email"),
  birthDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, "Date must be YYYY-MM-DD"),
  gender: z.enum(["man", "woman", "non_binary"]).optional(),
  languages: z.array(z.string().length(2, "Use ISO 639-1 codes")).optional(),
  latitude: z.number().min(-90).max(90).optional(),
  longitude: z.number().min(-180).max(180).optional(),
  searchRadius: z.number().min(1).max(500).optional(),
});

// Schema for updating a user (all fields optional)
export const updateUserSchema = z.object({
  firstName: z.string().min(1).optional(),
  lastName: z.string().min(1).optional(),
  email: z.string().email().optional(),
  birthDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).optional(),
  gender: z.enum(["man", "woman", "non_binary"]).optional(),
  languages: z.array(z.string().length(2, "Use ISO 639-1 codes")).optional(),
  latitude: z.number().min(-90).max(90).nullable().optional(),
  longitude: z.number().min(-180).max(180).nullable().optional(),
  searchRadius: z.number().min(1).max(500).optional(),
});

export type CreateUserInput = z.infer<typeof createUserSchema>;
export type UpdateUserInput = z.infer<typeof updateUserSchema>;

