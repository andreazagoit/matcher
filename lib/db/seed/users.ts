import { db } from "../drizzle";
import { users } from "../../models/users/schema";
import type { NewUser } from "../../models/users/schema";

function toUsername(name: string): string {
  return name.toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "").slice(0, 30);
}

export const SEED_USERS: Omit<NewUser, "id" | "emailVerified" | "createdAt" | "updatedAt">[] = [
  { name: "Admin System", email: "admin@matcher.local", birthdate: "1990-01-01", gender: "man" },
  {
    name: "Mario Rossi", email: "mario.rossi@example.com", birthdate: "1995-03-15", gender: "man",
    sexualOrientation: ["straight"], heightCm: 180,
    relationshipIntent: ["serious_relationship"], relationshipStyle: "monogamous",
    hasChildren: "no", wantsChildren: "yes",
    smoking: "never", drinking: "sometimes", activityLevel: "active", religion: "none",
    jobTitle: "Ingegnere software", educationLevel: "master", schoolName: "Politecnico di Milano",
    languages: ["italian", "english"],
  },
  {
    name: "Laura Bianchi", email: "laura.bianchi@example.com", birthdate: "1998-07-22", gender: "woman",
    sexualOrientation: ["bisexual"], heightCm: 165,
    relationshipIntent: ["serious_relationship", "casual_dating"], relationshipStyle: "ethical_non_monogamous",
    hasChildren: "no", wantsChildren: "open",
    smoking: "sometimes", drinking: "sometimes", activityLevel: "moderate", religion: "spiritual",
    jobTitle: "Grafica freelance", educationLevel: "bachelor", schoolName: "NABA Milano",
    languages: ["italian", "english", "french"],
  },
  {
    name: "Alessandro Verdi", email: "alessandro.verdi@example.com", birthdate: "1992-11-08", gender: "man",
    sexualOrientation: ["gay"], heightCm: 175,
    relationshipIntent: ["casual_dating"], relationshipStyle: "open",
    hasChildren: "no", wantsChildren: "no",
    smoking: "never", drinking: "regularly", activityLevel: "light", religion: "none",
    jobTitle: "Architetto", educationLevel: "master", schoolName: "Politecnico di Torino",
    languages: ["italian", "english", "spanish"],
  },
  {
    name: "Giulia Neri", email: "giulia.neri@example.com", birthdate: "1996-05-30", gender: "woman",
    sexualOrientation: ["straight"], heightCm: 170,
    relationshipIntent: ["serious_relationship"], relationshipStyle: "monogamous",
    hasChildren: "no", wantsChildren: "yes",
    smoking: "never", drinking: "sometimes", activityLevel: "very_active", religion: "christian",
    jobTitle: "Medico", educationLevel: "phd", schoolName: "Università degli Studi di Bologna",
    languages: ["italian", "english"],
  },
  {
    name: "Marco Ferrari", email: "marco.ferrari@example.com", birthdate: "1994-01-20", gender: "man",
    sexualOrientation: ["straight"], heightCm: 183,
    relationshipIntent: ["friendship"], relationshipStyle: "monogamous",
    hasChildren: "yes", wantsChildren: "no",
    smoking: "regularly", drinking: "regularly", activityLevel: "sedentary", religion: "none",
    jobTitle: "Commercialista", educationLevel: "bachelor",
    languages: ["italian"],
  },
  {
    name: "Sofia Romano", email: "sofia.romano@example.com", birthdate: "1997-09-12", gender: "woman",
    sexualOrientation: ["lesbian"], heightCm: 163,
    relationshipIntent: ["serious_relationship"], relationshipStyle: "monogamous",
    hasChildren: "no", wantsChildren: "yes",
    smoking: "never", drinking: "never", activityLevel: "active", religion: "none",
    jobTitle: "Insegnante", educationLevel: "master",
    languages: ["italian", "english", "german"],
  },
  {
    name: "Luca Colombo", email: "luca.colombo@example.com", birthdate: "1993-06-25", gender: "man",
    sexualOrientation: ["straight"], heightCm: 178,
    relationshipIntent: ["casual_dating"], relationshipStyle: "open",
    hasChildren: "no", wantsChildren: "open",
    smoking: "sometimes", drinking: "sometimes", activityLevel: "moderate", religion: "none",
    jobTitle: "Product manager", educationLevel: "bachelor",
    languages: ["italian", "english"],
  },
  {
    name: "Emma Ricci", email: "emma.ricci@example.com", birthdate: "1999-02-14", gender: "woman",
    sexualOrientation: ["pansexual"], heightCm: 168,
    relationshipIntent: ["serious_relationship", "casual_dating"], relationshipStyle: "ethical_non_monogamous",
    hasChildren: "no", wantsChildren: "open",
    smoking: "never", drinking: "sometimes", activityLevel: "active", religion: "spiritual",
    jobTitle: "Fotografa", educationLevel: "bachelor",
    languages: ["italian", "english", "french"],
  },
  {
    name: "Andrea Marino", email: "andrea.marino@example.com", birthdate: "1991-12-03", gender: "man",
    sexualOrientation: ["straight"], heightCm: 176,
    relationshipIntent: ["serious_relationship"], relationshipStyle: "monogamous",
    hasChildren: "yes", wantsChildren: "no",
    smoking: "never", drinking: "sometimes", activityLevel: "moderate", religion: "christian",
    jobTitle: "Avvocato", educationLevel: "master",
    languages: ["italian", "english"],
  },
  {
    name: "Chiara Greco", email: "chiara.greco@example.com", birthdate: "1996-08-18", gender: "woman",
    sexualOrientation: ["straight"], heightCm: 167,
    relationshipIntent: ["casual_dating"], relationshipStyle: "monogamous",
    hasChildren: "no", wantsChildren: "yes",
    smoking: "never", drinking: "sometimes", activityLevel: "active", religion: "none",
    jobTitle: "UX designer", educationLevel: "bachelor",
    languages: ["italian", "english", "spanish"],
  },
  {
    name: "Francesco Bruno", email: "francesco.bruno@example.com", birthdate: "1994-04-07", gender: "man",
    jobTitle: "Chef", educationLevel: "vocational",
    languages: ["italian", "english"],
  },
  {
    name: "Valentina Gallo", email: "valentina.gallo@example.com", birthdate: "1998-10-29", gender: "woman",
    jobTitle: "Biologa", educationLevel: "phd",
    languages: ["italian", "english", "german"],
  },
  {
    name: "Matteo Conti", email: "matteo.conti@example.com", birthdate: "1992-07-11", gender: "man",
    jobTitle: "Giornalista", educationLevel: "bachelor",
    languages: ["italian", "english"],
  },
  {
    name: "Alessia Costa", email: "alessia.costa@example.com", birthdate: "1997-03-22", gender: "woman",
    jobTitle: "Ricercatrice", educationLevel: "phd",
    languages: ["italian", "english", "french"],
  },
  {
    name: "Davide Fontana", email: "davide.fontana@example.com", birthdate: "1995-11-05", gender: "man",
    jobTitle: "Sviluppatore mobile", educationLevel: "bachelor",
    languages: ["italian", "english"],
  },
  {
    name: "Martina Caruso", email: "martina.caruso@example.com", birthdate: "1999-01-16", gender: "woman",
    jobTitle: "Studentessa universitaria", educationLevel: "bachelor",
    languages: ["italian", "english", "spanish"],
  },
  {
    name: "Simone Mancini", email: "simone.mancini@example.com", birthdate: "1993-09-28", gender: "man",
    jobTitle: "Consulente finanziario", educationLevel: "master",
    languages: ["italian", "english", "german"],
  },
  {
    name: "Giorgia Rizzo", email: "giorgia.rizzo@example.com", birthdate: "1996-12-09", gender: "woman",
    jobTitle: "Psicologa", educationLevel: "master",
    languages: ["italian", "english"],
  },
  {
    name: "Riccardo Lombardi", email: "riccardo.lombardi@example.com", birthdate: "1994-05-19", gender: "man",
    jobTitle: "Ingegnere civile", educationLevel: "master",
    languages: ["italian", "english"],
  },
  {
    name: "Elisa Moretti", email: "elisa.moretti@example.com", birthdate: "1998-08-31", gender: "woman",
    jobTitle: "Social media manager", educationLevel: "bachelor",
    languages: ["italian", "english", "french"],
  },
  {
    name: "Lorenzo Serra", email: "lorenzo.serra@example.com", birthdate: "1993-02-14", gender: "man",
    jobTitle: "Fisioterapista", educationLevel: "bachelor",
    languages: ["italian", "english"],
  },
  {
    name: "Francesca De Luca", email: "francesca.deluca@example.com", birthdate: "1997-06-07", gender: "woman",
    jobTitle: "Traduttrice", educationLevel: "master",
    languages: ["italian", "english", "french", "spanish"],
  },
  {
    name: "Tommaso Longo", email: "tommaso.longo@example.com", birthdate: "1995-10-23", gender: "man",
    jobTitle: "Musicista", educationLevel: "bachelor",
    languages: ["italian", "english"],
  },
  {
    name: "Beatrice Leone", email: "beatrice.leone@example.com", birthdate: "1998-04-11", gender: "woman",
    jobTitle: "Stylist", educationLevel: "vocational",
    languages: ["italian", "english", "french"],
  },
  {
    name: "Gabriele Martinelli", email: "gabriele.martinelli@example.com", birthdate: "1992-12-28", gender: "man",
    jobTitle: "Veterinario", educationLevel: "phd",
    languages: ["italian", "english"],
  },
];

export async function seedUsers() {
  console.log(`\n👤 Seeding ${SEED_USERS.length} users...`);

  const created: Record<string, { id: string; email: string }> = {};

  for (const seed of SEED_USERS) {
    const [user] = await db
      .insert(users)
      .values({ ...seed, username: toUsername(seed.name!) })
      .returning({ id: users.id, email: users.email });

    created[user.email] = user;
    console.log(`  ✓ ${seed.name}`);
  }

  console.log(`  → ${Object.keys(created).length} users created`);
  return created;
}
