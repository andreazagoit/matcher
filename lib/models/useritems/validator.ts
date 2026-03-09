import { z } from "zod";

export const MIN_PHOTOS = 4;
export const MAX_PHOTOS = 9;
export const MIN_PROMPTS = 2;
export const MAX_PROMPTS = 3;

export const photoItemSchema = z.object({
  type: z.literal("photo"),
  content: z.string().url("URL immagine non valido"),
  displayOrder: z.number().int().min(0).optional(),
});

export const promptItemSchema = z.object({
  type: z.literal("prompt"),
  promptKey: z.string().min(1, "Seleziona un prompt"),
  content: z.string().min(1, "La risposta non può essere vuota").max(300, "Massimo 300 caratteri"),
  displayOrder: z.number().int().min(0).optional(),
});

export type PhotoItemInput = z.infer<typeof photoItemSchema>;
export type PromptItemInput = z.infer<typeof promptItemSchema>;
