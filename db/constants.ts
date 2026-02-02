// Valori personali selezionabili
export const VALUES_OPTIONS = [
  "famiglia",
  "carriera",
  "amicizia",
  "avventura",
  "stabilità",
  "creatività",
  "spiritualità",
  "salute",
  "libertà",
  "onestà",
  "lealtà",
  "ambizione",
  "empatia",
  "rispetto",
  "crescita personale",
] as const;

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

export type Value = (typeof VALUES_OPTIONS)[number];
export type Interest = (typeof INTERESTS_OPTIONS)[number];

