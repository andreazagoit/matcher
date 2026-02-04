import { z } from "zod";

/**
 * Validators per Users
 * 
 * NUOVA ARCHITETTURA:
 * - Users: solo dati anagrafici base
 * - Values/Interests: ora fanno parte del sistema test (tests/types.ts)
 */

// Schema per creare un utente (solo dati base)
export const createUserSchema = z.object({
  firstName: z.string().min(1, "First name is required"),
  lastName: z.string().min(1, "Last name is required"),
  email: z.string().email("Invalid email"),
  birthDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, "Date must be YYYY-MM-DD"),
  gender: z.enum(["man", "woman", "non_binary"]).optional(),
});

// Schema per aggiornare un utente (tutti i campi opzionali)
export const updateUserSchema = z.object({
  firstName: z.string().min(1).optional(),
  lastName: z.string().min(1).optional(),
  email: z.string().email().optional(),
  birthDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).optional(),
  gender: z.enum(["man", "woman", "non_binary"]).optional(),
});

export type CreateUserInput = z.infer<typeof createUserSchema>;
export type UpdateUserInput = z.infer<typeof updateUserSchema>;

