import { db } from "../drizzle";
import { profileItems } from "../../models/profileitems/schema";

const SAMPLE_PHOTO_URL = "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=800&q=80";
const SAMPLE_PHOTO_URLS = [
  "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=800&q=80",
  "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=800&q=80",
  "https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?w=800&q=80",
  "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=800&q=80",
  "https://images.unsplash.com/photo-1504257432389-52343af06ae3?w=800&q=80",
  "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=800&q=80",
];

type CardInput = {
  type: "photo" | "prompt";
  promptKey?: string;
  content: string;
  displayOrder: number;
};

const CARD_TEMPLATES: CardInput[][] = [
  // Template A — outgoing/creative
  [
    { type: "photo", content: SAMPLE_PHOTO_URLS[0], displayOrder: 0 },
    { type: "prompt", promptKey: "controversial_opinion", content: "Il caffè nel pomeriggio non è sbagliato. Sono pronto a difendere questa posizione.", displayOrder: 1 },
    { type: "photo", content: SAMPLE_PHOTO_URLS[1], displayOrder: 2 },
    { type: "prompt", promptKey: "sunday_plan", content: "Mercato del sabato mattina, lettura al sole e cena improvvisata con quello che trovo in frigo.", displayOrder: 3 },
    { type: "photo", content: SAMPLE_PHOTO_URLS[2], displayOrder: 4 },
    { type: "photo", content: SAMPLE_PHOTO_URLS[3], displayOrder: 5 },
  ],
  // Template B — introspective/romantic
  [
    { type: "photo", content: SAMPLE_PHOTO_URLS[1], displayOrder: 0 },
    { type: "prompt", promptKey: "cant_live_without", content: "Le playlist curate con cura. Una colonna sonora sbagliata può rovinare qualsiasi momento.", displayOrder: 1 },
    { type: "photo", content: SAMPLE_PHOTO_URLS[2], displayOrder: 2 },
    { type: "photo", content: SAMPLE_PHOTO_URLS[3], displayOrder: 3 },
    { type: "prompt", promptKey: "win_me_over", content: "Proponi un piano spontaneo senza troppe aspettative. O preparami qualcosa da mangiare.", displayOrder: 4 },
    { type: "photo", content: SAMPLE_PHOTO_URLS[4], displayOrder: 5 },
  ],
  // Template C — humorous
  [
    { type: "photo", content: SAMPLE_PHOTO_URLS[2], displayOrder: 0 },
    { type: "photo", content: SAMPLE_PHOTO_URLS[0], displayOrder: 1 },
    { type: "prompt", promptKey: "innocent_red_flag", content: "Riorganizzare la libreria degli altri senza chiedere permesso.", displayOrder: 2 },
    { type: "photo", content: SAMPLE_PHOTO_URLS[4], displayOrder: 3 },
    { type: "prompt", promptKey: "guilty_pleasure", content: "Guardare tutorial di cucina senza mai cucinare niente di quello che vedo.", displayOrder: 4 },
    { type: "photo", content: SAMPLE_PHOTO_URLS[5], displayOrder: 5 },
  ],
];

export async function seedProfileCards(userIds: string[]) {
  console.log(`\n🃏 Seeding profile cards for ${userIds.length} users...`);

  let count = 0;
  for (let i = 0; i < userIds.length; i++) {
    const userId = userIds[i];
    const template = CARD_TEMPLATES[i % CARD_TEMPLATES.length];

    await db.insert(profileItems).values(
      template.map((card) => ({
        userId,
        type: card.type,
        promptKey: card.promptKey ?? null,
        content: card.content,
        displayOrder: card.displayOrder,
      }))
    );
    count += template.length;
  }

  console.log(`  → ${count} profile cards created`);
}
