import "dotenv/config";
import { db } from "../drizzle";
import { users } from "../../models/users/schema";
import { assessments, type AssessmentAnswersJson } from "../../models/assessments/schema";
import { profiles } from "../../models/profiles/schema";
import { QUESTIONS, SECTIONS, ASSESSMENT_NAME } from "../../models/assessments/questions";
import { assembleProfile } from "../../models/assessments/assembler";
import { generateAllUserEmbeddings } from "../../embeddings";
import { createSpace } from "../../models/spaces/operations";

/**
 * Seed - Crea 25 utenti con test completati e profili
 */

const SEED_USERS = [
  // Admin user
  { firstName: "Admin", lastName: "System", email: "admin@matcher.local", birthDate: "1990-01-01", gender: "man" as const },
  // Regular users
  { firstName: "Mario", lastName: "Rossi", email: "mario.rossi@example.com", birthDate: "1995-03-15", gender: "man" as const },
  { firstName: "Laura", lastName: "Bianchi", email: "laura.bianchi@example.com", birthDate: "1998-07-22", gender: "woman" as const },
  { firstName: "Alessandro", lastName: "Verdi", email: "alessandro.verdi@example.com", birthDate: "1992-11-08", gender: "man" as const },
  { firstName: "Giulia", lastName: "Neri", email: "giulia.neri@example.com", birthDate: "1996-05-30", gender: "woman" as const },
  { firstName: "Marco", lastName: "Ferrari", email: "marco.ferrari@example.com", birthDate: "1994-01-20", gender: "man" as const },
  { firstName: "Sofia", lastName: "Romano", email: "sofia.romano@example.com", birthDate: "1997-09-12", gender: "woman" as const },
  { firstName: "Luca", lastName: "Colombo", email: "luca.colombo@example.com", birthDate: "1993-06-25", gender: "man" as const },
  { firstName: "Emma", lastName: "Ricci", email: "emma.ricci@example.com", birthDate: "1999-02-14", gender: "woman" as const },
  { firstName: "Andrea", lastName: "Marino", email: "andrea.marino@example.com", birthDate: "1991-12-03", gender: "man" as const },
  { firstName: "Chiara", lastName: "Greco", email: "chiara.greco@example.com", birthDate: "1996-08-18", gender: "woman" as const },
  { firstName: "Francesco", lastName: "Bruno", email: "francesco.bruno@example.com", birthDate: "1994-04-07", gender: "man" as const },
  { firstName: "Valentina", lastName: "Gallo", email: "valentina.gallo@example.com", birthDate: "1998-10-29", gender: "woman" as const },
  { firstName: "Matteo", lastName: "Conti", email: "matteo.conti@example.com", birthDate: "1992-07-11", gender: "man" as const },
  { firstName: "Alessia", lastName: "Costa", email: "alessia.costa@example.com", birthDate: "1997-03-22", gender: "woman" as const },
  { firstName: "Davide", lastName: "Fontana", email: "davide.fontana@example.com", birthDate: "1995-11-05", gender: "man" as const },
  { firstName: "Martina", lastName: "Caruso", email: "martina.caruso@example.com", birthDate: "1999-01-16", gender: "woman" as const },
  { firstName: "Simone", lastName: "Mancini", email: "simone.mancini@example.com", birthDate: "1993-09-28", gender: "man" as const },
  { firstName: "Giorgia", lastName: "Rizzo", email: "giorgia.rizzo@example.com", birthDate: "1996-12-09", gender: "woman" as const },
  { firstName: "Riccardo", lastName: "Lombardi", email: "riccardo.lombardi@example.com", birthDate: "1994-05-19", gender: "man" as const },
  { firstName: "Elisa", lastName: "Moretti", email: "elisa.moretti@example.com", birthDate: "1998-08-31", gender: "woman" as const },
  { firstName: "Lorenzo", lastName: "Serra", email: "lorenzo.serra@example.com", birthDate: "1993-02-14", gender: "man" as const },
  { firstName: "Francesca", lastName: "De Luca", email: "francesca.deluca@example.com", birthDate: "1997-06-07", gender: "woman" as const },
  { firstName: "Tommaso", lastName: "Longo", email: "tommaso.longo@example.com", birthDate: "1995-10-23", gender: "man" as const },
  { firstName: "Beatrice", lastName: "Leone", email: "beatrice.leone@example.com", birthDate: "1998-04-11", gender: "woman" as const },
  { firstName: "Gabriele", lastName: "Martinelli", email: "gabriele.martinelli@example.com", birthDate: "1992-12-28", gender: "man" as const },
];

// Risposte aperte di esempio
const OPEN_ANSWERS: Record<string, string[]> = {
  "psy-open": [
    "Sono una persona curiosa e riflessiva, mi piace ascoltare gli altri",
    "Mi considero empatico/a e attento/a ai dettagli",
    "Sono spontaneo/a e mi piace vivere il momento",
    "Mi definisco una persona calma e razionale",
    "Sono creativo/a e sempre alla ricerca di nuove idee",
  ],
  "val-open": [
    "Cerco sempre di essere autentico/a e fedele a me stesso/a",
    "L'onestà e il rispetto sono fondamentali per me",
    "Credo nell'equilibrio tra lavoro e vita personale",
    "La famiglia e gli affetti sono la mia priorità",
    "Voglio fare la differenza e aiutare gli altri",
  ],
  "int-open": [
    "Amo la natura, il trekking e la fotografia",
    "Mi appassiona la musica, suono la chitarra da anni",
    "Adoro viaggiare e scoprire nuove culture",
    "Mi piace cucinare e sperimentare ricette nuove",
    "Sono appassionato/a di cinema e serie TV",
  ],
  "beh-open": [
    "All'inizio sono riservato/a ma poi mi apro molto",
    "Mi piace costruire connessioni profonde gradualmente",
    "Sono diretto/a e apprezzo chi lo è con me",
    "Preferisco poche relazioni ma significative",
    "Sono molto affettuoso/a quando mi sento a mio agio",
  ],
};

/**
 * Genera risposte casuali per il test
 * Formato: { questionId: valore }
 * - Chiuse: 1-5
 * - Aperte: stringa
 */
function generateRandomAnswers(): AssessmentAnswersJson {
  const answers: AssessmentAnswersJson = {};

  for (const section of SECTIONS) {
    for (const question of QUESTIONS[section]) {
      if (question.type === "closed") {
        // Valore casuale 1-5
        answers[question.id] = Math.floor(Math.random() * 5) + 1;
      } else {
        // Risposta aperta casuale
        const openOptions = OPEN_ANSWERS[question.id] || [""];
        answers[question.id] = openOptions[Math.floor(Math.random() * openOptions.length)];
      }
    }
  }

  return answers;
}

async function seed() {
  console.log(`🌱 Seeding database with ${SEED_USERS.length} users...\n`);

  if (!process.env.OPENAI_API_KEY) {
    console.error("❌ OPENAI_API_KEY is required");
    process.exit(1);
  }

  try {
    // 0. Create system admin and Space
    const adminData = SEED_USERS[0];
    const [adminUser] = await db.insert(users).values(adminData).returning();
    console.log(`  🔑 Created Admin User: ${adminData.email}`);

    const systemSpace = await createSpace({
      name: "Matcher System",
      slug: "matcher-system",
      description: "Official Matcher System Space for internal use",
      redirectUris: [
        "http://localhost:3000/api/auth/callback/matcher", // Auth.js callback
        "http://localhost:3000/oauth/callback",
        "http://localhost:3000/dashboard/oauth-test-callback",
      ],
      creatorId: adminUser.id,
      visibility: "hidden",
      joinPolicy: "invite_only",
      image: "https://placehold.co/400x400/333/fff?text=System",
    });
    console.log(`  🔑 Created System Space:`);
    console.log(`     Client ID: ${systemSpace.clientId}`);
    console.log(`     Secret Key: ${systemSpace.secretKey}`);
    console.log(`     Secret Key: ${systemSpace.secretKey}`);
    console.log(`     ⚠️  Add these to .env as OAUTH_CLIENT_ID and OAUTH_CLIENT_SECRET`);

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

    // Process other users (skip admin)
    for (let i = 1; i < SEED_USERS.length; i++) {
      const userData = SEED_USERS[i];

      // 1. Crea user
      const [user] = await db.insert(users).values(userData).returning();

      // 2. Genera risposte test
      const answers = generateRandomAnswers();

      // 3. Salva assessment
      await db.insert(assessments).values({
        userId: user.id,
        assessmentName: ASSESSMENT_NAME,
        answers,
        status: "completed",
      });

      // 4. Assembla ProfileData (ora restituisce semplici stringhe)
      const profileData = assembleProfile(answers);

      // 5. Genera embeddings
      const embeddings = await generateAllUserEmbeddings({
        psychological: profileData.psychologicalDesc,
        values: profileData.valuesDesc,
        interests: profileData.interestsDesc,
        behavioral: profileData.behavioralDesc,
      });

      // 6. Crea profilo
      await db.insert(profiles).values({
        userId: user.id,
        psychologicalDesc: profileData.psychologicalDesc,
        valuesDesc: profileData.valuesDesc,
        interestsDesc: profileData.interestsDesc,
        behavioralDesc: profileData.behavioralDesc,
        psychologicalEmbedding: embeddings.psychological,
        valuesEmbedding: embeddings.values,
        interestsEmbedding: embeddings.interests,
        behavioralEmbedding: embeddings.behavioral,
        assessmentVersion: 1,
      });

      console.log(`  ✓ ${i + 1}/${SEED_USERS.length} - ${userData.firstName} ${userData.lastName}`);
    }

    console.log(`\n✅ Created ${SEED_USERS.length} users with test sessions and profiles`);

  } catch (error) {
    console.error("❌ Seed failed:", error);
    process.exit(1);
  }

  process.exit(0);
}

seed();
