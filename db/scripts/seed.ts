import { db, users } from "../index";
import type { Value, Interest } from "../constants";

const seedUsers: {
  firstName: string;
  lastName: string;
  email: string;
  birthDate: string;
  values: Value[];
  interests: Interest[];
}[] = [
  {
    firstName: "Mario",
    lastName: "Rossi",
    email: "mario.rossi@example.com",
    birthDate: "1995-03-15",
    values: ["famiglia", "onestà", "ambizione"],
    interests: ["sport", "viaggi", "cucina"],
  },
  {
    firstName: "Laura",
    lastName: "Bianchi",
    email: "laura.bianchi@example.com",
    birthDate: "1998-07-22",
    values: ["creatività", "libertà", "empatia"],
    interests: ["musica", "arte", "yoga"],
  },
  {
    firstName: "Alessandro",
    lastName: "Verdi",
    email: "alessandro.verdi@example.com",
    birthDate: "1992-11-08",
    values: ["carriera", "crescita personale", "stabilità"],
    interests: ["tecnologia", "cinema", "escursionismo"],
  },
  {
    firstName: "Giulia",
    lastName: "Neri",
    email: "giulia.neri@example.com",
    birthDate: "1996-05-30",
    values: ["avventura", "amicizia", "rispetto"],
    interests: ["fotografia", "lettura", "viaggi"],
  },
];

async function seed() {
  console.log("🌱 Seeding database...");

  try {
    await db.insert(users).values(seedUsers);
    console.log(`✅ Inserted ${seedUsers.length} users`);
  } catch (error) {
    console.error("❌ Seed failed:", error);
    process.exit(1);
  }

  console.log("✅ Seed completed!");
  process.exit(0);
}

seed();
