import { z } from "zod";
import { categorySchema, coordinatesSchema } from "@/lib/models/shared/validator";
import { attendeeStatusEnum } from "./schema";

export const createEventSchema = z.object({
  spaceId: z.string().uuid(),
  title: z.string().min(1).max(200),
  description: z.string().max(2000).optional(),
  location: z.string().max(300).optional(),
  coordinates: coordinatesSchema.optional(),
  startsAt: z.coerce.date(),
  endsAt: z.coerce.date().optional(),
  cover: z.url(),
  images: z.array(z.url()).max(20).default([]),
  categories: z.array(categorySchema).max(10).default([]),
  maxAttendees: z.number().int().positive().optional(),
  price: z.number().int().nonnegative().optional(),
  currency: z.string().length(3).toLowerCase().default("eur"),
});

export const updateEventSchema = z.object({
  title: z.string().min(1).max(200).optional(),
  description: z.string().max(2000).optional(),
  location: z.string().max(300).optional(),
  coordinates: coordinatesSchema.nullable().optional(),
  startsAt: z.coerce.date().optional(),
  endsAt: z.coerce.date().optional(),
  cover: z.url().optional(),
  images: z.array(z.url()).max(20).optional(),
  categories: z.array(categorySchema).max(10).optional(),
  maxAttendees: z.number().int().positive().optional(),
  price: z.number().int().nonnegative().optional(),
  currency: z.string().length(3).toLowerCase().optional(),
});

export const respondToEventSchema = z.object({
  eventId: z.string().uuid(),
  status: z.enum(attendeeStatusEnum.enumValues),
});

export type CreateEventInput = z.infer<typeof createEventSchema>;
export type UpdateEventInput = z.infer<typeof updateEventSchema>;
export type RespondToEventInput = z.infer<typeof respondToEventSchema>;
