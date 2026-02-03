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

export type Value = (typeof VALUES_OPTIONS)[number];

