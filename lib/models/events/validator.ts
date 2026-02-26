import { z } from "zod";
import { ALL_TAGS } from "@/lib/models/tags/data";
import { attendeeStatusEnum } from "./schema";

const tagSchema = z.enum(ALL_TAGS as [string, ...string[]]);

const coordinatesSchema = z.object({
  lat: z.number().min(-90).max(90),
  lon: z.number().min(-180).max(180),
});

export const createEventSchema = z.object({
  spaceId: z.string().uuid(),
  title: z.string().min(1).max(200),
  description: z.string().max(2000).optional(),
  location: z.string().max(300).optional(),
  coordinates: coordinatesSchema.optional(),
  startsAt: z.coerce.date(),
  endsAt: z.coerce.date().optional(),
  maxAttendees: z.number().int().positive().optional(),
  tags: z.array(tagSchema).max(10).default([]),
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
  maxAttendees: z.number().int().positive().optional(),
  tags: z.array(tagSchema).max(10).optional(),
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
