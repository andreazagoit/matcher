import { z } from "zod";
import { categorySchema } from "@/lib/models/shared/validator";

const visibilityValues = ["public", "private", "hidden"] as const;
const joinPolicyValues = ["open", "apply", "invite_only"] as const;

export const createSpaceSchema = z.object({
  name: z.string().min(1).max(100),
  slug: z.string().min(1).max(100),
  description: z.string().max(1000).optional(),
  cover: z.url(),
  images: z.array(z.url()).max(20).default([]),
  categories: z.array(categorySchema).max(10).default([]),
  visibility: z.enum(visibilityValues).default("public"),
  joinPolicy: z.enum(joinPolicyValues).default("open"),
});

export const updateSpaceSchema = z.object({
  name: z.string().min(1).max(100).optional(),
  description: z.string().max(1000).optional(),
  cover: z.url().optional(),
  images: z.array(z.url()).max(20).optional(),
  categories: z.array(categorySchema).max(10).optional(),
  visibility: z.enum(visibilityValues).optional(),
  joinPolicy: z.enum(joinPolicyValues).optional(),
});

export type CreateSpaceInput = z.infer<typeof createSpaceSchema>;
export type UpdateSpaceInput = z.infer<typeof updateSpaceSchema>;
