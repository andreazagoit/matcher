// Interessi selezionabili
export const INTERESTS_OPTIONS = [
  "sport",
  "musica",
  "viaggi",
  "cucina",
  "arte",
  "cinema",
  "lettura",
  "fotografia",
  "tecnologia",
  "gaming",
  "natura",
  "fitness",
  "yoga",
  "danza",
  "teatro",
  "moda",
  "animali",
  "volontariato",
  "escursionismo",
  "meditazione",
] as const;

export type Interest = (typeof INTERESTS_OPTIONS)[number];

