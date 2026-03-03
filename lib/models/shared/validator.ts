import { z } from "zod";
import { CATEGORIES } from "@/lib/models/categories/data";

export const SUPPORTED_LANGUAGES = [
    "italian", "english", "french", "spanish", "german", "portuguese",
    "arabic", "chinese", "japanese", "russian", "hindi", "turkish", "other",
] as const;

export const categorySchema = z.enum(CATEGORIES as unknown as [string, ...string[]]);

export const coordinatesSchema = z.object({
    lat: z.number().min(-90).max(90),
    lon: z.number().min(-180).max(180),
});

export type CoordinatesInput = z.infer<typeof coordinatesSchema>;
