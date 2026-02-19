import { TAG_CATEGORIES, ALL_TAGS, isValidTag } from "./data";

/**
 * Get all tag categories and their associated tags.
 */
export function getTagCategories() {
  return Object.entries(TAG_CATEGORIES).map(([category, tags]) => ({
    category,
    tags,
  }));
}

/**
 * Get all valid tags as a flat list.
 */
export function getAllTags() {
  return ALL_TAGS;
}

/**
 * Validate a list of tags.
 */
export function validateTags(tags: string[]): string | null {
  return tags.find((t) => !isValidTag(t)) || null;
}
