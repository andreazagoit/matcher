import { z } from "zod";
import { ALL_TAGS } from "@/lib/models/tags/data";

const tagSchema = z.enum(ALL_TAGS as [string, ...string[]]);

const visibilityValues = ["public", "private", "hidden"] as const;
const joinPolicyValues = ["open", "apply", "invite_only"] as const;

export const createSpaceSchema = z.object({
  name: z.string().min(1).max(100),
  description: z.string().max(1000).optional(),
  image: z.string().url().optional(),
  visibility: z.enum(visibilityValues).default("public"),
  joinPolicy: z.enum(joinPolicyValues).default("open"),
  tags: z.array(tagSchema).max(10).default([]),
});

export const updateSpaceSchema = z.object({
  name: z.string().min(1).max(100).optional(),
  description: z.string().max(1000).optional(),
  image: z.string().url().optional(),
  visibility: z.enum(visibilityValues).optional(),
  joinPolicy: z.enum(joinPolicyValues).optional(),
  isActive: z.boolean().optional(),
  tags: z.array(tagSchema).max(10).optional(),
});

export const spacesByTagsSchema = z.object({
  tags: z.array(tagSchema).min(1),
  matchAll: z.boolean().default(false),
});

export type CreateSpaceInput = z.infer<typeof createSpaceSchema>;
export type UpdateSpaceInput = z.infer<typeof updateSpaceSchema>;
export type SpacesByTagsInput = z.infer<typeof spacesByTagsSchema>;
