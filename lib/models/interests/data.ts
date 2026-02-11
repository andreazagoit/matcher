// Selectable interest options (standard tags for GraphQL enums)
export const INTERESTS_OPTIONS = [
  "SPORTS",
  "MUSIC",
  "TRAVEL",
  "COOKING",
  "ART",
  "CINEMA",
  "READING",
  "PHOTOGRAPHY",
  "TECHNOLOGY",
  "GAMING",
  "NATURE",
  "FITNESS",
  "YOGA",
  "DANCE",
  "THEATER",
  "FASHION",
  "ANIMALS",
  "VOLUNTEERING",
  "HIKING",
  "MEDITATION",
] as const;

export type Interest = (typeof INTERESTS_OPTIONS)[number];
