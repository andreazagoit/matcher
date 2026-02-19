import { db } from "../drizzle";
import { users } from "../../models/users/schema";

export const SEED_USERS = [
  { givenName: "Admin", familyName: "System", email: "admin@matcher.local", birthdate: "1990-01-01", gender: "man" as const },
  { givenName: "Mario", familyName: "Rossi", email: "mario.rossi@example.com", birthdate: "1995-03-15", gender: "man" as const },
  { givenName: "Laura", familyName: "Bianchi", email: "laura.bianchi@example.com", birthdate: "1998-07-22", gender: "woman" as const },
  { givenName: "Alessandro", familyName: "Verdi", email: "alessandro.verdi@example.com", birthdate: "1992-11-08", gender: "man" as const },
  { givenName: "Giulia", familyName: "Neri", email: "giulia.neri@example.com", birthdate: "1996-05-30", gender: "woman" as const },
  { givenName: "Marco", familyName: "Ferrari", email: "marco.ferrari@example.com", birthdate: "1994-01-20", gender: "man" as const },
  { givenName: "Sofia", familyName: "Romano", email: "sofia.romano@example.com", birthdate: "1997-09-12", gender: "woman" as const },
  { givenName: "Luca", familyName: "Colombo", email: "luca.colombo@example.com", birthdate: "1993-06-25", gender: "man" as const },
  { givenName: "Emma", familyName: "Ricci", email: "emma.ricci@example.com", birthdate: "1999-02-14", gender: "woman" as const },
  { givenName: "Andrea", familyName: "Marino", email: "andrea.marino@example.com", birthdate: "1991-12-03", gender: "man" as const },
  { givenName: "Chiara", familyName: "Greco", email: "chiara.greco@example.com", birthdate: "1996-08-18", gender: "woman" as const },
  { givenName: "Francesco", familyName: "Bruno", email: "francesco.bruno@example.com", birthdate: "1994-04-07", gender: "man" as const },
  { givenName: "Valentina", familyName: "Gallo", email: "valentina.gallo@example.com", birthdate: "1998-10-29", gender: "woman" as const },
  { givenName: "Matteo", familyName: "Conti", email: "matteo.conti@example.com", birthdate: "1992-07-11", gender: "man" as const },
  { givenName: "Alessia", familyName: "Costa", email: "alessia.costa@example.com", birthdate: "1997-03-22", gender: "woman" as const },
  { givenName: "Davide", familyName: "Fontana", email: "davide.fontana@example.com", birthdate: "1995-11-05", gender: "man" as const },
  { givenName: "Martina", familyName: "Caruso", email: "martina.caruso@example.com", birthdate: "1999-01-16", gender: "woman" as const },
  { givenName: "Simone", familyName: "Mancini", email: "simone.mancini@example.com", birthdate: "1993-09-28", gender: "man" as const },
  { givenName: "Giorgia", familyName: "Rizzo", email: "giorgia.rizzo@example.com", birthdate: "1996-12-09", gender: "woman" as const },
  { givenName: "Riccardo", familyName: "Lombardi", email: "riccardo.lombardi@example.com", birthdate: "1994-05-19", gender: "man" as const },
  { givenName: "Elisa", familyName: "Moretti", email: "elisa.moretti@example.com", birthdate: "1998-08-31", gender: "woman" as const },
  { givenName: "Lorenzo", familyName: "Serra", email: "lorenzo.serra@example.com", birthdate: "1993-02-14", gender: "man" as const },
  { givenName: "Francesca", familyName: "De Luca", email: "francesca.deluca@example.com", birthdate: "1997-06-07", gender: "woman" as const },
  { givenName: "Tommaso", familyName: "Longo", email: "tommaso.longo@example.com", birthdate: "1995-10-23", gender: "man" as const },
  { givenName: "Beatrice", familyName: "Leone", email: "beatrice.leone@example.com", birthdate: "1998-04-11", gender: "woman" as const },
  { givenName: "Gabriele", familyName: "Martinelli", email: "gabriele.martinelli@example.com", birthdate: "1992-12-28", gender: "man" as const },
];

export type SeedUser = (typeof SEED_USERS)[number];

/**
 * Seed all users. Returns the created DB rows indexed by email.
 */
export async function seedUsers() {
  console.log(`\n👤 Seeding ${SEED_USERS.length} users...`);

  const created: Record<string, { id: string; email: string }> = {};

  for (const seed of SEED_USERS) {
    const [user] = await db
      .insert(users)
      .values({
        ...seed,
        name: `${seed.givenName} ${seed.familyName}`,
      })
      .returning({ id: users.id, email: users.email });

    created[user.email] = user;
    console.log(`  ✓ ${seed.givenName} ${seed.familyName}`);
  }

  console.log(`  → ${Object.keys(created).length} users created`);
  return created;
}
