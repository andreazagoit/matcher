import { z } from "zod";
import { categorySchema } from "@/lib/models/shared/validator";

const visibilityValues = ["public", "private", "hidden"] as const;
const joinPolicyValues = ["open", "apply", "invite_only"] as const;

export const createSpaceSchema = z.object({
  name: z.string().min(1).max(100),
  description: z.string().max(1000).optional(),
  image: z.url().optional(),
  visibility: z.enum(visibilityValues).default("public"),
  joinPolicy: z.enum(joinPolicyValues).default("open"),
  categories: z.array(categorySchema).max(10).default([]),
});

export const updateSpaceSchema = z.object({
  name: z.string().min(1).max(100).optional(),
  description: z.string().max(1000).optional(),
  image: z.url().optional(),
  visibility: z.enum(visibilityValues).optional(),
  joinPolicy: z.enum(joinPolicyValues).optional(),
  isActive: z.boolean().optional(),
  categories: z.array(categorySchema).max(10).optional(),
});

export const spacesByCategoriesSchema = z.object({
  categories: z.array(categorySchema).min(1),
  matchAll: z.boolean().default(false),
});

export type CreateSpaceInput = z.infer<typeof createSpaceSchema>;
export type UpdateSpaceInput = z.infer<typeof updateSpaceSchema>;
export type SpacesByCategoriesInput = z.infer<typeof spacesByCategoriesSchema>;
