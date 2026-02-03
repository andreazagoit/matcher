import { z } from "zod";
import { VALUES_OPTIONS } from "../values/data";
import { INTERESTS_OPTIONS } from "../interests/data";

// Schema per Value enum
export const valueSchema = z.enum(VALUES_OPTIONS as unknown as [string, ...string[]]);

// Schema per Interest enum
export const interestSchema = z.enum(INTERESTS_OPTIONS as unknown as [string, ...string[]]);

// Schema per creare un utente
export const createUserSchema = z.object({
  firstName: z.string().min(1, "First name is required"),
  lastName: z.string().min(1, "Last name is required"),
  email: z.email(),
  birthDate: z.string(),
  values: z
    .array(valueSchema)
    .min(1, "At least one value is required")
    .max(10, "Maximum 10 values allowed"),
  interests: z
    .array(interestSchema)
    .min(1, "At least one interest is required")
    .max(10, "Maximum 10 interests allowed"),
});

// Schema per aggiornare un utente (tutti i campi opzionali)
export const updateUserSchema = z.object({
  firstName: z.string().min(1).optional(),
  lastName: z.string().min(1).optional(),
  email: z.email().optional(),
  birthDate: z.string().optional(),
  values: z
    .array(valueSchema)
    .min(1, "At least one value is required")
    .max(10, "Maximum 10 values allowed")
    .optional(),
  interests: z
    .array(interestSchema)
    .min(1, "At least one interest is required")
    .max(10, "Maximum 10 interests allowed")
    .optional(),
});

export type CreateUserInput = z.infer<typeof createUserSchema>;
export type UpdateUserInput = z.infer<typeof updateUserSchema>;

