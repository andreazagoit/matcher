// Valori personali selezionabili (tag standard per GraphQL enum)
export const VALUES_OPTIONS = [
  "FAMILY",
  "CAREER",
  "FRIENDSHIP",
  "ADVENTURE",
  "STABILITY",
  "CREATIVITY",
  "SPIRITUALITY",
  "HEALTH",
  "FREEDOM",
  "HONESTY",
  "LOYALTY",
  "AMBITION",
  "EMPATHY",
  "RESPECT",
  "PERSONAL_GROWTH",
] as const;

export type Value = (typeof VALUES_OPTIONS)[number];
