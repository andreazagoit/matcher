import "dotenv/config";
import { db } from "../drizzle";
import { users } from "../../models/users/schema";
import { createSpace } from "../../models/spaces/operations";

/**
 * Seeding Script - Creates users and spaces.
 *
 * Note: Assessments, profiles, and embeddings are managed by Identity Matcher.
 * Run the identitymatcher seed script to populate that data.
 */

const SEED_USERS = [
  // Admin user
  { givenName: "Admin", familyName: "System", email: "admin@matcher.local", birthdate: "1990-01-01", gender: "man" as const },
  // Regular users
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

async function seed() {
  console.log(`🌱 Seeding database with ${SEED_USERS.length} users...\n`);

  try {
    // 1. Create system admin and Space
    const adminData = {
      ...SEED_USERS[0],
      name: `${SEED_USERS[0].givenName} ${SEED_USERS[0].familyName}`,
    };
    const [adminUser] = await db.insert(users).values(adminData).returning();
    console.log(`  🔑 Created Admin User: ${adminData.email}`);

    const { space: systemSpace } = await createSpace({
      name: "Matcher System",
      slug: "matcher-system",
      description: "Official Matcher System Space for internal use",
      creatorId: adminUser.id,
      visibility: "hidden",
      joinPolicy: "invite_only",
      image: "https://placehold.co/400x400/333/fff?text=System",
    });
    console.log(`  🔑 Created System Space: ${systemSpace.slug}`);

    // Create some public demo spaces
    await createSpace({
      name: "Tech Innovators",
      slug: "tech-innovators",
      description: "A community for technology enthusiasts and innovators.",
      creatorId: adminUser.id,
      visibility: "public",
      image: "https://placehold.co/400x400/2563eb/fff?text=Tech",
    });
    console.log(`  🚀 Created Space: Tech Innovators`);

    await createSpace({
      name: "Nature Lovers",
      slug: "nature-lovers",
      description: "Hiking, photography, and outdoor adventures.",
      creatorId: adminUser.id,
      visibility: "public",
      image: "https://placehold.co/400x400/16a34a/fff?text=Nature",
    });
    console.log(`  🌲 Created Space: Nature Lovers`);

    // 2. Create other users (skip admin)
    for (let i = 1; i < SEED_USERS.length; i++) {
      const userData = {
        ...SEED_USERS[i],
        name: `${SEED_USERS[i].givenName} ${SEED_USERS[i].familyName}`,
      };

      await db.insert(users).values(userData).returning();
      console.log(`  ✓ ${i + 1}/${SEED_USERS.length} - ${userData.givenName} ${userData.familyName}`);
    }

    console.log(`\n✅ Created ${SEED_USERS.length} users and demo spaces`);
    console.log(`\nℹ️  Note: Assessments, profiles, and embeddings are managed by Identity Matcher.`);
    console.log(`   Run the identitymatcher seed script to populate that data.`);

  } catch (error) {
    console.error("❌ Seed failed:", error);
    process.exit(1);
  }

  process.exit(0);
}

seed();
