/**
 * Shared category vocabulary used across Events and Spaces.
 *
 * Categories are plain strings (normalized, lowercase, no spaces).
 * They replace the old tag system — users declare interest via
 * impressions (action: 'liked') rather than a profile array.
 */

export const CATEGORIES: string[] = [
  "sport",
  "outdoor",
  "music",
  "art",
  "food",
  "travel",
  "wellness",
  "tech",
  "culture",
  "cinema",
  "social",
  "animals",
  "fashion",
  "sustainability",
  "entrepreneurship",
  "science",
  "spirituality",
  "volunteering",
  "nightlife",
  "photography",
  "dance",
  "crafts",
  "languages",
  "comedy",
];

const CATEGORY_SET = new Set(CATEGORIES);

export function isValidCategory(category: string): boolean {
  return CATEGORY_SET.has(category);
}
