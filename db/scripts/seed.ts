import type { Value, Interest } from "../constants";
import { createUser } from "../users";

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
  {
    firstName: "Marco",
    lastName: "Ferrari",
    email: "marco.ferrari@example.com",
    birthDate: "1994-01-20",
    values: ["salute", "spiritualità", "rispetto"],
    interests: ["fitness", "escursionismo", "meditazione"],
  },
  {
    firstName: "Sofia",
    lastName: "Romano",
    email: "sofia.romano@example.com",
    birthDate: "1997-09-12",
    values: ["creatività", "libertà", "empatia"],
    interests: ["arte", "teatro", "danza"],
  },
  {
    firstName: "Luca",
    lastName: "Colombo",
    email: "luca.colombo@example.com",
    birthDate: "1993-06-25",
    values: ["carriera", "ambizione", "crescita personale"],
    interests: ["tecnologia", "gaming", "cinema"],
  },
  {
    firstName: "Emma",
    lastName: "Ricci",
    email: "emma.ricci@example.com",
    birthDate: "1999-02-14",
    values: ["empatia", "onestà", "rispetto"],
    interests: ["volontariato", "animali", "lettura"],
  },
  {
    firstName: "Andrea",
    lastName: "Marino",
    email: "andrea.marino@example.com",
    birthDate: "1991-12-03",
    values: ["avventura", "libertà", "crescita personale"],
    interests: ["viaggi", "escursionismo", "fotografia"],
  },
  {
    firstName: "Chiara",
    lastName: "Greco",
    email: "chiara.greco@example.com",
    birthDate: "1996-08-18",
    values: ["spiritualità", "salute", "rispetto"],
    interests: ["yoga", "meditazione", "natura"],
  },
  {
    firstName: "Francesco",
    lastName: "Bruno",
    email: "francesco.bruno@example.com",
    birthDate: "1994-04-07",
    values: ["famiglia", "lealtà", "stabilità"],
    interests: ["cucina", "lettura", "cinema"],
  },
  {
    firstName: "Valentina",
    lastName: "Gallo",
    email: "valentina.gallo@example.com",
    birthDate: "1998-10-29",
    values: ["creatività", "libertà", "empatia"],
    interests: ["moda", "arte", "fotografia"],
  },
  {
    firstName: "Matteo",
    lastName: "Conti",
    email: "matteo.conti@example.com",
    birthDate: "1992-07-11",
    values: ["carriera", "ambizione", "crescita personale"],
    interests: ["tecnologia", "gaming", "cinema"],
  },
  {
    firstName: "Alessia",
    lastName: "Costa",
    email: "alessia.costa@example.com",
    birthDate: "1997-03-22",
    values: ["salute", "stabilità", "rispetto"],
    interests: ["fitness", "yoga", "cucina"],
  },
  {
    firstName: "Davide",
    lastName: "Fontana",
    email: "davide.fontana@example.com",
    birthDate: "1995-11-05",
    values: ["creatività", "libertà", "empatia"],
    interests: ["musica", "teatro", "cinema"],
  },
  {
    firstName: "Martina",
    lastName: "Caruso",
    email: "martina.caruso@example.com",
    birthDate: "1999-01-16",
    values: ["amicizia", "lealtà", "empatia"],
    interests: ["musica", "danza", "teatro"],
  },
  {
    firstName: "Simone",
    lastName: "Mancini",
    email: "simone.mancini@example.com",
    birthDate: "1993-09-28",
    values: ["ambizione", "salute", "rispetto"],
    interests: ["sport", "fitness", "natura"],
  },
  {
    firstName: "Giorgia",
    lastName: "Rizzo",
    email: "giorgia.rizzo@example.com",
    birthDate: "1996-12-09",
    values: ["avventura", "libertà", "crescita personale"],
    interests: ["viaggi", "lettura", "fotografia"],
  },
  {
    firstName: "Riccardo",
    lastName: "Lombardi",
    email: "riccardo.lombardi@example.com",
    birthDate: "1994-05-19",
    values: ["onestà", "lealtà", "rispetto"],
    interests: ["sport", "cinema", "lettura"],
  },
  {
    firstName: "Elisa",
    lastName: "Moretti",
    email: "elisa.moretti@example.com",
    birthDate: "1998-08-31",
    values: ["creatività", "libertà", "empatia"],
    interests: ["arte", "fotografia", "teatro"],
  },
  {
    firstName: "Lorenzo",
    lastName: "Serra",
    email: "lorenzo.serra@example.com",
    birthDate: "1993-02-14",
    values: ["famiglia", "onestà", "stabilità"],
    interests: ["cucina", "cinema", "lettura"],
  },
  {
    firstName: "Francesca",
    lastName: "De Luca",
    email: "francesca.deluca@example.com",
    birthDate: "1997-06-07",
    values: ["carriera", "ambizione", "crescita personale"],
    interests: ["tecnologia", "cinema", "lettura"],
  },
  {
    firstName: "Tommaso",
    lastName: "Longo",
    email: "tommaso.longo@example.com",
    birthDate: "1995-10-23",
    values: ["avventura", "libertà", "amicizia"],
    interests: ["viaggi", "escursionismo", "fotografia"],
  },
  {
    firstName: "Beatrice",
    lastName: "Leone",
    email: "beatrice.leone@example.com",
    birthDate: "1998-04-11",
    values: ["creatività", "empatia", "rispetto"],
    interests: ["arte", "teatro", "danza"],
  },
  {
    firstName: "Gabriele",
    lastName: "Martinelli",
    email: "gabriele.martinelli@example.com",
    birthDate: "1992-12-28",
    values: ["spiritualità", "salute", "rispetto"],
    interests: ["yoga", "meditazione", "natura"],
  },
  {
    firstName: "Sara",
    lastName: "Vitale",
    email: "sara.vitale@example.com",
    birthDate: "1996-07-19",
    values: ["empatia", "onestà", "lealtà"],
    interests: ["volontariato", "animali", "lettura"],
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
