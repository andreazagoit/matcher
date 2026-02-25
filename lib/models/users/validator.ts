import { z } from "zod";
import {
  genderEnum,
  sexualOrientationEnum,
  relationshipIntentEnum,
  relationshipStyleEnum,
  hasChildrenEnum,
  wantsChildrenEnum,
  religionEnum,
  smokingEnum,
  drinkingEnum,
  activityLevelEnum,
  educationLevelEnum,
  ethnicityEnum,
} from "./schema";

export const SUPPORTED_LANGUAGES = [
  "italian", "english", "french", "spanish", "german", "portuguese",
  "arabic", "chinese", "japanese", "russian", "hindi", "turkish", "other",
] as const;

const usernameSchema = z
  .string()
  .min(3, "Username must be at least 3 characters")
  .max(30, "Username must be at most 30 characters")
  .regex(/^[a-z0-9_]+$/, "Username can only contain lowercase letters, numbers and underscores");

export const createUserSchema = z.object({
  username: usernameSchema,
  name: z.string().min(1, "Name is required"),
  email: z.string().email("Invalid email"),
  birthdate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, "Date must be YYYY-MM-DD"),
  gender: z.enum(genderEnum.enumValues).optional(),
});

export const signUpSchema = createUserSchema.extend({
  gender: z.enum(genderEnum.enumValues, { message: "Seleziona il genere" }),
});

export const updateUserSchema = z.object({
  username: usernameSchema.optional(),
  name: z.string().min(1).optional(),
  email: z.string().email().optional(),
  birthdate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).optional(),
  gender: z.enum(genderEnum.enumValues).optional(),

  // Orientation & identity
  sexualOrientation: z.array(z.enum(sexualOrientationEnum.enumValues)).optional(),
  heightCm: z.number().int().min(100).max(250).optional(),

  // Relational intent
  relationshipIntent: z.array(z.enum(relationshipIntentEnum.enumValues)).optional(),
  relationshipStyle: z.enum(relationshipStyleEnum.enumValues).optional(),
  hasChildren: z.enum(hasChildrenEnum.enumValues).optional(),
  wantsChildren: z.enum(wantsChildrenEnum.enumValues).optional(),

  // Lifestyle
  religion: z.enum(religionEnum.enumValues).optional(),
  smoking: z.enum(smokingEnum.enumValues).optional(),
  drinking: z.enum(drinkingEnum.enumValues).optional(),
  activityLevel: z.enum(activityLevelEnum.enumValues).optional(),

  // Identity & background
  jobTitle: z.string().max(100).optional(),
  educationLevel: z.enum(educationLevelEnum.enumValues).optional(),
  schoolName: z.string().max(150).optional(),
  languages: z.array(z.enum(SUPPORTED_LANGUAGES)).optional(),
  ethnicity: z.enum(ethnicityEnum.enumValues).optional(),
  locationText: z.string().max(255).optional(),
});

/**
 * Represents all data collected across the multi-step sign-up form.
 * heightCm is a string because HTML inputs always return strings;
 * the page converts it to number before calling the API.
 */
export const signupFormSchema = z.object({
  name: z.string(),
  birthdate: z.string(),
  gender: z.enum(genderEnum.enumValues).optional(),
  sexualOrientation: z.array(z.enum(sexualOrientationEnum.enumValues)).default([]),
  relationshipIntent: z.array(z.enum(relationshipIntentEnum.enumValues)).default([]),
  relationshipStyle: z.enum(relationshipStyleEnum.enumValues).optional(),
  hasChildren: z.enum(hasChildrenEnum.enumValues).optional(),
  wantsChildren: z.enum(wantsChildrenEnum.enumValues).optional(),
  smoking: z.enum(smokingEnum.enumValues).optional(),
  drinking: z.enum(drinkingEnum.enumValues).optional(),
  activityLevel: z.enum(activityLevelEnum.enumValues).optional(),
  religion: z.enum(religionEnum.enumValues).optional(),
  heightCm: z.string().optional(),
  jobTitle: z.string().max(100).optional(),
  educationLevel: z.enum(educationLevelEnum.enumValues).optional(),
  schoolName: z.string().max(150).optional(),
  languages: z.array(z.enum(SUPPORTED_LANGUAGES)).default([]),
  ethnicity: z.enum(ethnicityEnum.enumValues).optional(),
  initialInterests: z.array(z.string()).default([]),
  username: usernameSchema,
  email: z.string().email(),
});

export type CreateUserInput = z.infer<typeof createUserSchema>;
export type UpdateUserInput = z.infer<typeof updateUserSchema>;
export type SignupFormData = z.infer<typeof signupFormSchema>;
