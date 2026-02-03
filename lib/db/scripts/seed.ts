import "dotenv/config";
import type { Value } from "@/lib/models/values/operations";
import type { Interest } from "@/lib/models/interests/operations";
import { createUser } from "../../models/users/operations";

// 25 utenti statici per test
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
    values: ["FAMILY", "HONESTY", "AMBITION"],
    interests: ["SPORTS", "TRAVEL", "COOKING"],
  },
  {
    firstName: "Laura",
    lastName: "Bianchi",
    email: "laura.bianchi@example.com",
    birthDate: "1998-07-22",
    values: ["CREATIVITY", "FREEDOM", "EMPATHY"],
    interests: ["MUSIC", "ART", "YOGA"],
  },
  {
    firstName: "Alessandro",
    lastName: "Verdi",
    email: "alessandro.verdi@example.com",
    birthDate: "1992-11-08",
    values: ["CAREER", "PERSONAL_GROWTH", "STABILITY"],
    interests: ["TECHNOLOGY", "CINEMA", "HIKING"],
  },
  {
    firstName: "Giulia",
    lastName: "Neri",
    email: "giulia.neri@example.com",
    birthDate: "1996-05-30",
    values: ["ADVENTURE", "FRIENDSHIP", "RESPECT"],
    interests: ["PHOTOGRAPHY", "READING", "TRAVEL"],
  },
  {
    firstName: "Marco",
    lastName: "Ferrari",
    email: "marco.ferrari@example.com",
    birthDate: "1994-01-20",
    values: ["HEALTH", "SPIRITUALITY", "RESPECT"],
    interests: ["FITNESS", "HIKING", "MEDITATION"],
  },
  {
    firstName: "Sofia",
    lastName: "Romano",
    email: "sofia.romano@example.com",
    birthDate: "1997-09-12",
    values: ["CREATIVITY", "FREEDOM", "EMPATHY"],
    interests: ["ART", "THEATER", "DANCE"],
  },
  {
    firstName: "Luca",
    lastName: "Colombo",
    email: "luca.colombo@example.com",
    birthDate: "1993-06-25",
    values: ["CAREER", "AMBITION", "PERSONAL_GROWTH"],
    interests: ["TECHNOLOGY", "GAMING", "CINEMA"],
  },
  {
    firstName: "Emma",
    lastName: "Ricci",
    email: "emma.ricci@example.com",
    birthDate: "1999-02-14",
    values: ["EMPATHY", "HONESTY", "RESPECT"],
    interests: ["VOLUNTEERING", "ANIMALS", "READING"],
  },
  {
    firstName: "Andrea",
    lastName: "Marino",
    email: "andrea.marino@example.com",
    birthDate: "1991-12-03",
    values: ["ADVENTURE", "FREEDOM", "PERSONAL_GROWTH"],
    interests: ["TRAVEL", "HIKING", "PHOTOGRAPHY"],
  },
  {
    firstName: "Chiara",
    lastName: "Greco",
    email: "chiara.greco@example.com",
    birthDate: "1996-08-18",
    values: ["SPIRITUALITY", "HEALTH", "RESPECT"],
    interests: ["YOGA", "MEDITATION", "NATURE"],
  },
  {
    firstName: "Francesco",
    lastName: "Bruno",
    email: "francesco.bruno@example.com",
    birthDate: "1994-04-07",
    values: ["FAMILY", "LOYALTY", "STABILITY"],
    interests: ["COOKING", "READING", "CINEMA"],
  },
  {
    firstName: "Valentina",
    lastName: "Gallo",
    email: "valentina.gallo@example.com",
    birthDate: "1998-10-29",
    values: ["CREATIVITY", "FREEDOM", "EMPATHY"],
    interests: ["FASHION", "ART", "PHOTOGRAPHY"],
  },
  {
    firstName: "Matteo",
    lastName: "Conti",
    email: "matteo.conti@example.com",
    birthDate: "1992-07-11",
    values: ["CAREER", "AMBITION", "PERSONAL_GROWTH"],
    interests: ["TECHNOLOGY", "GAMING", "CINEMA"],
  },
  {
    firstName: "Alessia",
    lastName: "Costa",
    email: "alessia.costa@example.com",
    birthDate: "1997-03-22",
    values: ["HEALTH", "STABILITY", "RESPECT"],
    interests: ["FITNESS", "YOGA", "COOKING"],
  },
  {
    firstName: "Davide",
    lastName: "Fontana",
    email: "davide.fontana@example.com",
    birthDate: "1995-11-05",
    values: ["CREATIVITY", "FREEDOM", "EMPATHY"],
    interests: ["MUSIC", "THEATER", "CINEMA"],
  },
  {
    firstName: "Martina",
    lastName: "Caruso",
    email: "martina.caruso@example.com",
    birthDate: "1999-01-16",
    values: ["FRIENDSHIP", "LOYALTY", "EMPATHY"],
    interests: ["MUSIC", "DANCE", "THEATER"],
  },
  {
    firstName: "Simone",
    lastName: "Mancini",
    email: "simone.mancini@example.com",
    birthDate: "1993-09-28",
    values: ["AMBITION", "HEALTH", "RESPECT"],
    interests: ["SPORTS", "FITNESS", "NATURE"],
  },
  {
    firstName: "Giorgia",
    lastName: "Rizzo",
    email: "giorgia.rizzo@example.com",
    birthDate: "1996-12-09",
    values: ["ADVENTURE", "FREEDOM", "PERSONAL_GROWTH"],
    interests: ["TRAVEL", "READING", "PHOTOGRAPHY"],
  },
  {
    firstName: "Riccardo",
    lastName: "Lombardi",
    email: "riccardo.lombardi@example.com",
    birthDate: "1994-05-19",
    values: ["HONESTY", "LOYALTY", "RESPECT"],
    interests: ["SPORTS", "CINEMA", "READING"],
  },
  {
    firstName: "Elisa",
    lastName: "Moretti",
    email: "elisa.moretti@example.com",
    birthDate: "1998-08-31",
    values: ["CREATIVITY", "FREEDOM", "EMPATHY"],
    interests: ["ART", "PHOTOGRAPHY", "THEATER"],
  },
  {
    firstName: "Lorenzo",
    lastName: "Serra",
    email: "lorenzo.serra@example.com",
    birthDate: "1993-02-14",
    values: ["FAMILY", "HONESTY", "STABILITY"],
    interests: ["COOKING", "CINEMA", "READING"],
  },
  {
    firstName: "Francesca",
    lastName: "De Luca",
    email: "francesca.deluca@example.com",
    birthDate: "1997-06-07",
    values: ["CAREER", "AMBITION", "PERSONAL_GROWTH"],
    interests: ["TECHNOLOGY", "CINEMA", "READING"],
  },
  {
    firstName: "Tommaso",
    lastName: "Longo",
    email: "tommaso.longo@example.com",
    birthDate: "1995-10-23",
    values: ["ADVENTURE", "FREEDOM", "FRIENDSHIP"],
    interests: ["TRAVEL", "HIKING", "PHOTOGRAPHY"],
  },
  {
    firstName: "Beatrice",
    lastName: "Leone",
    email: "beatrice.leone@example.com",
    birthDate: "1998-04-11",
    values: ["CREATIVITY", "EMPATHY", "RESPECT"],
    interests: ["ART", "THEATER", "DANCE"],
  },
  {
    firstName: "Gabriele",
    lastName: "Martinelli",
    email: "gabriele.martinelli@example.com",
    birthDate: "1992-12-28",
    values: ["SPIRITUALITY", "HEALTH", "RESPECT"],
    interests: ["YOGA", "MEDITATION", "NATURE"],
  },
  {
    firstName: "Sara",
    lastName: "Vitale",
    email: "sara.vitale@example.com",
    birthDate: "1996-07-19",
    values: ["EMPATHY", "HONESTY", "LOYALTY"],
    interests: ["VOLUNTEERING", "ANIMALS", "READING"],
  },
];

async function seed() {
  console.log("🌱 Seeding database with 25 users...");

  if (!process.env.OPENAI_API_KEY) {
    console.error("❌ OPENAI_API_KEY is required for seeding");
    process.exit(1);
  }

  try {
    // Usa la stessa funzione createUser dei resolver
    console.log("📊 Creating users with embeddings...");
    for (let i = 0; i < seedUsers.length; i++) {
      const user = seedUsers[i];
      await createUser(user);
      if ((i + 1) % 5 === 0) {
        console.log(`  ✓ Created ${i + 1}/${seedUsers.length} users`);
      }
    }

    console.log(`✅ Inserted ${seedUsers.length} users with embeddings`);
  } catch (error) {
    console.error("❌ Seed failed:", error);
    process.exit(1);
  }

  console.log("✅ Seed completed!");
  process.exit(0);
}

seed();
