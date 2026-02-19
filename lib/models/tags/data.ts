/**
 * Shared tag vocabulary used across Profiles, Events, and Spaces.
 *
 * Tags are plain strings. Localization is handled at the UI layer.
 */

export const TAG_CATEGORIES: Record<string, string[]> = {
  outdoor: [
    "trekking",
    "camping",
    "climbing",
    "cycling",
    "beach",
    "mountains",
    "gardening",
  ],
  culture: [
    "cinema",
    "theater",
    "live_music",
    "museums",
    "reading",
    "photography",
    "art",
  ],
  food: [
    "cooking",
    "restaurants",
    "wine",
    "craft_beer",
    "street_food",
    "coffee",
  ],
  sports: [
    "running",
    "gym",
    "yoga",
    "swimming",
    "football",
    "tennis",
    "padel",
    "basketball",
  ],
  creative: [
    "music",
    "drawing",
    "writing",
    "diy",
    "gaming",
    "coding",
  ],
  social: [
    "travel",
    "volunteering",
    "languages",
    "pets",
    "parties",
    "board_games",
  ],
};

// ─── Flat list and validation ───────────────────────────────────────

export const ALL_TAGS: string[] = Object.values(TAG_CATEGORIES).flat();

const TAG_SET = new Set(ALL_TAGS);

export function isValidTag(tag: string): boolean {
  return TAG_SET.has(tag);
}
