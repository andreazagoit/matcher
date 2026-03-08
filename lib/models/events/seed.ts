import { spaces } from "../spaces/schema";
import { db } from "../../db/drizzle";
import { createEvent } from "./operations";
import type { CreateEventInput } from "./validator";

type SeedEvent = Omit<CreateEventInput, "createdBy" | "spaceId" | "currency" | "coordinates" | "startsAt" | "endsAt"> & {
  spaceSlug: string;
  lat: number;
  lon: number;
  startsAt: Date;
  endsAt?: Date;
};

const now = new Date();
const d = (daysFromNow: number, hour = 18, minute = 0) => {
  const date = new Date(now);
  date.setDate(date.getDate() + daysFromNow);
  date.setHours(hour, minute, 0, 0);
  return date;
};

const SEED_EVENTS: SeedEvent[] = [
  // sport
  {
    spaceSlug: "fitness-partners",
    title: "Running in Parco Sempione",
    description: "Corsa di gruppo al mattino nel parco. Tutti i ritmi benvenuti!",
    location: "Parco Sempione, Milano",
    lat: 45.4754, lon: 9.1742,
    startsAt: d(30, 7, 30), endsAt: d(30, 9, 0),
    cover: "https://images.unsplash.com/photo-1571008887538-b36bb32f4571?w=800&q=80",
    images: [],
    maxAttendees: 25,
    categories: ["sport", "outdoor"],
  },
  // outdoor
  {
    spaceSlug: "outdoor-adventures",
    title: "Trekking al Monte Generoso",
    description: "Escursione di media difficoltà con vista panoramica su Lombardia e Svizzera.",
    location: "Monte Generoso, Lombardia",
    lat: 45.9319, lon: 9.0197,
    startsAt: d(38, 8, 0), endsAt: d(38, 17, 0),
    cover: "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=800&q=80",
    images: [],
    maxAttendees: 15,
    categories: ["outdoor", "sport"],
  },
  {
    spaceSlug: "outdoor-adventures",
    title: "Ciclismo sul Lago di Garda",
    description: "Giro in bici lungo la sponda bresciana del Lago di Garda. ~40km.",
    location: "Desenzano del Garda, Brescia",
    lat: 45.4664, lon: 10.5351,
    startsAt: d(50, 9, 0), endsAt: d(50, 14, 0),
    cover: "https://images.unsplash.com/photo-1541625602330-2277a4c46182?w=800&q=80",
    images: [],
    maxAttendees: 20,
    categories: ["outdoor", "sport"],
  },
  // music
  {
    spaceSlug: "music-live",
    title: "Concerto Jazz al Blue Note",
    description: "Serata jazz con musicisti emergenti. Cena opzionale inclusa.",
    location: "Blue Note Milano",
    lat: 45.4891, lon: 9.1973,
    startsAt: d(45, 21, 0), endsAt: d(45, 24, 0),
    cover: "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=800&q=80",
    images: [],
    maxAttendees: 50,
    categories: ["music", "nightlife"],
  },
  // art
  {
    spaceSlug: "artists-creatives",
    title: "Open Studio a Isola",
    description: "Visita agli atelier degli artisti del quartiere Isola.",
    location: "Quartiere Isola, Milano",
    lat: 45.4890, lon: 9.1908,
    startsAt: d(48, 15, 0), endsAt: d(48, 20, 0),
    cover: "https://images.unsplash.com/photo-1513364776144-60967b0f800f?w=800&q=80",
    images: [],
    maxAttendees: 30,
    categories: ["art", "photography"],
  },
  // food
  {
    spaceSlug: "foodies-wine",
    title: "Degustazione Vini della Toscana",
    description: "Serata di degustazione con sommelier professionista. 6 etichette selezionate.",
    location: "Enoteca Pinchiorri, Firenze",
    lat: 43.7711, lon: 11.2600,
    startsAt: d(55, 19, 30), endsAt: d(55, 22, 30),
    cover: "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=800&q=80",
    images: [],
    maxAttendees: 16,
    categories: ["food", "social"],
  },
  {
    spaceSlug: "foodies-wine",
    title: "Cooking Class: Pasta Fresca",
    description: "Impariamo a fare la pasta fresca all'uovo con uno chef professionista.",
    location: "Scuola di Cucina Eataly, Milano",
    lat: 45.4641, lon: 9.2026,
    startsAt: d(65, 11, 0), endsAt: d(65, 14, 30),
    cover: "https://images.unsplash.com/photo-1504754524776-8f4f37790ca0?w=800&q=80",
    images: [],
    maxAttendees: 12,
    categories: ["food", "social"],
  },
  // social / dating
  {
    spaceSlug: "milano-singles",
    title: "Aperitivo al Naviglio Grande",
    description: "Aperitivo di gruppo sui Navigli, ottimo per conoscere nuove persone.",
    location: "Naviglio Grande, Milano",
    lat: 45.4545, lon: 9.1718,
    startsAt: d(32, 19, 0), endsAt: d(32, 22, 0),
    cover: "https://images.unsplash.com/photo-1517732306149-e8f829eb588a?w=800&q=80",
    images: [],
    maxAttendees: 20,
    categories: ["travel", "social"],
  },
  {
    spaceSlug: "roma-dating",
    title: "Tour al Colosseo al tramonto",
    description: "Visita guidata al Colosseo durante il tramonto. Atmosfera unica.",
    location: "Colosseo, Roma",
    lat: 41.8902, lon: 12.4922,
    startsAt: d(70, 17, 30), endsAt: d(70, 20, 0),
    cover: "https://images.unsplash.com/photo-1555992336-03a23c7b20ee?w=800&q=80",
    images: [],
    maxAttendees: 15,
    categories: ["travel", "culture"],
  },
  // wellness
  {
    spaceSlug: "mind-body",
    title: "Yoga all'alba a Villa Borghese",
    description: "Sessione di yoga all'aperto al sorgere del sole. Porta il tuo tappetino.",
    location: "Villa Borghese, Roma",
    lat: 41.9136, lon: 12.4921,
    startsAt: d(42, 6, 30), endsAt: d(42, 8, 0),
    cover: "https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?w=800&q=80",
    images: [],
    maxAttendees: 20,
    categories: ["wellness", "sport", "spirituality"],
  },
  // tech
  {
    spaceSlug: "tech-innovators",
    title: "Hackathon AI Weekend",
    description: "48 ore per costruire un progetto AI. Team di 3-4 persone.",
    location: "Impact Hub, Milano",
    lat: 45.4669, lon: 9.1890,
    startsAt: d(90, 9, 0), endsAt: d(92, 17, 0),
    cover: "https://images.unsplash.com/photo-1518770660439-4636190af475?w=800&q=80",
    images: [],
    maxAttendees: 40,
    categories: ["tech", "entrepreneurship"],
  },
  {
    spaceSlug: "tech-professionals",
    title: "Meetup: LLM in produzione",
    description: "Talk tecnico su come portare modelli linguistici in produzione.",
    location: "Google for Startups Campus, Milano",
    lat: 45.4654, lon: 9.1856,
    startsAt: d(58, 18, 30), endsAt: d(58, 21, 0),
    cover: "https://images.unsplash.com/photo-1461749280684-dccba630e2f6?w=800&q=80",
    images: [],
    maxAttendees: 60,
    categories: ["tech", "science"],
  },
  // culture
  {
    spaceSlug: "book-culture-club",
    title: "Book Club: Il nome della rosa",
    description: "Discussione del capolavoro di Umberto Eco.",
    location: "Libreria Feltrinelli, Milano",
    lat: 45.4721, lon: 9.1651,
    startsAt: d(40, 18, 30), endsAt: d(40, 21, 0),
    cover: "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=800&q=80",
    images: [],
    maxAttendees: 15,
    categories: ["culture", "social"],
  },
  {
    spaceSlug: "book-culture-club",
    title: "Laboratorio di Scrittura Creativa",
    description: "Workshop pratico di scrittura creativa con un autore ospite.",
    location: "Centro Culturale di Milano",
    lat: 45.4641, lon: 9.1919,
    startsAt: d(80, 10, 0), endsAt: d(80, 13, 0),
    cover: "https://images.unsplash.com/photo-1455390582262-044cdead277a?w=800&q=80",
    images: [],
    maxAttendees: 12,
    categories: ["culture", "art"],
  },
  // cinema
  {
    spaceSlug: "cinema-lovers",
    title: "Serata Cinema Anteo",
    description: "Film + discussione post-proiezione. Un modo diverso per conoscersi.",
    location: "Anteo Palazzo del Cinema, Milano",
    lat: 45.4829, lon: 9.1786,
    startsAt: d(60, 20, 30), endsAt: d(60, 23, 30),
    cover: "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=800&q=80",
    images: [],
    maxAttendees: 25,
    categories: ["cinema", "social"],
  },
  // social / nightlife
  {
    spaceSlug: "milano-singles",
    title: "Speed Dating al Brera",
    description: "Classico speed dating nel quartiere Brera. 5 minuti per fare colpo!",
    location: "Caffè Letterario Brera, Milano",
    lat: 45.4722, lon: 9.1864,
    startsAt: d(45, 20, 0), endsAt: d(45, 23, 0),
    cover: "https://images.unsplash.com/photo-1529543544282-ea669407fca3?w=800&q=80",
    images: [],
    maxAttendees: 30,
    categories: ["social", "nightlife"],
  },
  {
    spaceSlug: "roma-dating",
    title: "Cena a Trastevere",
    description: "Cena romantica in gruppo nel cuore di Trastevere.",
    location: "Piazza di Santa Maria in Trastevere, Roma",
    lat: 41.8897, lon: 12.4699,
    startsAt: d(35, 20, 0), endsAt: d(35, 23, 30),
    cover: "https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=800&q=80",
    images: [],
    maxAttendees: 12,
    categories: ["social", "food", "nightlife"],
  },
  // dance
  {
    spaceSlug: "dance-move",
    title: "Salsa Night a Milano",
    description: "Serata di salsa cubana con lezione introduttiva per principianti.",
    location: "La Salumeria della Musica, Milano",
    lat: 45.4462, lon: 9.1895,
    startsAt: d(36, 21, 0), endsAt: d(36, 24, 0),
    cover: "https://images.unsplash.com/photo-1504609773096-104ff2c73ba4?w=800&q=80",
    images: [],
    maxAttendees: 40,
    categories: ["dance", "music", "social"],
  },
  // animals
  {
    spaceSlug: "pet-lovers",
    title: "Dog Social al Parco delle Cave",
    description: "Passeggiata social con i nostri amici a quattro zampe.",
    location: "Parco delle Cave, Milano",
    lat: 45.4280, lon: 9.1080,
    startsAt: d(33, 10, 0), endsAt: d(33, 12, 0),
    cover: "https://images.unsplash.com/photo-1450778869180-41d0601e046e?w=800&q=80",
    images: [],
    maxAttendees: 30,
    categories: ["animals", "outdoor", "social"],
  },
  // sustainability
  {
    spaceSlug: "green-community",
    title: "Pulizia Spiagge a Ostia",
    description: "Volontariato ambientale sulla spiaggia. Guanti e sacchi forniti.",
    location: "Lungomare di Ostia, Roma",
    lat: 41.7335, lon: 12.2388,
    startsAt: d(44, 9, 0), endsAt: d(44, 13, 0),
    cover: "https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?w=800&q=80",
    images: [],
    maxAttendees: 50,
    categories: ["sustainability", "volunteering", "outdoor"],
  },
  // languages
  {
    spaceSlug: "language-exchange",
    title: "Language Exchange Aperitivo",
    description: "Parla italiano, inglese, spagnolo e francese con madrelingua.",
    location: "Bar Brera, Milano",
    lat: 45.4722, lon: 9.1851,
    startsAt: d(37, 19, 0), endsAt: d(37, 21, 30),
    cover: "https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8?w=800&q=80",
    images: [],
    maxAttendees: 30,
    categories: ["languages", "social", "culture"],
  },
  // comedy
  {
    spaceSlug: "comedy-fun",
    title: "Stand-up Comedy Night",
    description: "Serata di stand-up con comici emergenti della scena milanese.",
    location: "The Comedy Club, Milano",
    lat: 45.4654, lon: 9.1922,
    startsAt: d(52, 21, 0), endsAt: d(52, 23, 30),
    cover: "https://images.unsplash.com/photo-1527224538127-2104bb71c51b?w=800&q=80",
    images: [],
    maxAttendees: 60,
    categories: ["comedy", "social", "nightlife"],
  },
  // fashion
  {
    spaceSlug: "style-fashion",
    title: "Vintage Market & Style Swap",
    description: "Scambia capi vintage, scopri nuovi stili e connettiti.",
    location: "BASE Milano",
    lat: 45.4497, lon: 9.1873,
    startsAt: d(62, 11, 0), endsAt: d(62, 18, 0),
    cover: "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800&q=80",
    images: [],
    maxAttendees: 80,
    categories: ["fashion", "art", "social"],
  },
  // entrepreneurship
  {
    spaceSlug: "tech-professionals",
    title: "Startup Pitch Night",
    description: "Presenta la tua idea a investitori e mentor. Networking post-evento.",
    location: "Talent Garden, Milano",
    lat: 45.5015, lon: 9.2197,
    startsAt: d(75, 18, 0), endsAt: d(75, 21, 0),
    cover: "https://images.unsplash.com/photo-1497366216548-37526070297c?w=800&q=80",
    images: [],
    maxAttendees: 80,
    categories: ["entrepreneurship", "tech", "social"],
  },
  // science
  {
    spaceSlug: "tech-innovators",
    title: "Science Café: Intelligenza Artificiale",
    description: "Talk divulgativo sull'AI per tutti. Q&A aperto.",
    location: "Museo della Scienza e Tecnologia, Milano",
    lat: 45.4635, lon: 9.1714,
    startsAt: d(68, 18, 0), endsAt: d(68, 20, 30),
    cover: "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b?w=800&q=80",
    images: [],
    maxAttendees: 100,
    categories: ["science", "tech", "culture"],
  },
  // photography
  {
    spaceSlug: "creators-hub",
    title: "Urban Photography Walk",
    description: "Passeggiata fotografica nei Navigli con feedback collettivo.",
    location: "Navigli, Milano",
    lat: 45.4500, lon: 9.1750,
    startsAt: d(43, 16, 0), endsAt: d(43, 19, 0),
    cover: "https://images.unsplash.com/photo-1452802447250-470a88ac82bc?w=800&q=80",
    images: [],
    maxAttendees: 20,
    categories: ["photography", "art", "outdoor"],
  },
  // crafts
  {
    spaceSlug: "creators-hub",
    title: "Workshop Ceramica",
    description: "Impara le basi della ceramica a mano con un artigiano esperto.",
    location: "Studio Artigiano, Milano",
    lat: 45.4780, lon: 9.2010,
    startsAt: d(56, 15, 0), endsAt: d(56, 18, 0),
    cover: "https://images.unsplash.com/photo-1565193566173-7a0ee3dbe261?w=800&q=80",
    images: [],
    maxAttendees: 12,
    categories: ["crafts", "art", "social"],
  },
  // volunteering
  {
    spaceSlug: "soul-service",
    title: "Mensa Solidale",
    description: "Volontariato alla mensa per persone senza dimora.",
    location: "Caritas Milano",
    lat: 45.4654, lon: 9.1920,
    startsAt: d(31, 11, 0), endsAt: d(31, 14, 0),
    cover: "https://images.unsplash.com/photo-1469571486292-0ba58a3f068b?w=800&q=80",
    images: [],
    maxAttendees: 20,
    categories: ["volunteering", "social"],
  },
  // spirituality
  {
    spaceSlug: "soul-service",
    title: "Meditazione Guidata al Tramonto",
    description: "Sessione di meditazione mindfulness all'aperto.",
    location: "Giardini Pubblici Indro Montanelli, Milano",
    lat: 45.4724, lon: 9.2006,
    startsAt: d(34, 19, 0), endsAt: d(34, 20, 30),
    cover: "https://images.unsplash.com/photo-1506126613408-eca07ce68773?w=800&q=80",
    images: [],
    maxAttendees: 25,
    categories: ["spirituality", "wellness"],
  },
];

export async function seedEvents(adminUserId: string) {
  console.log(`\n📅 Seeding ${SEED_EVENTS.length} events...`);

  const allSpaces = await db.select({ id: spaces.id, slug: spaces.slug }).from(spaces);
  const spaceBySlug = Object.fromEntries(allSpaces.map((s) => [s.slug, s.id]));

  let created = 0;
  let skipped = 0;

  for (const ev of SEED_EVENTS) {
    const spaceId = spaceBySlug[ev.spaceSlug];
    if (!spaceId) {
      console.log(`  ⚠️  Space not found: ${ev.spaceSlug}`);
      skipped++;
      continue;
    }

    try {
      await createEvent({
        spaceId,
        title: ev.title,
        description: ev.description,
        location: ev.location,
        coordinates: { lat: ev.lat, lon: ev.lon },
        startsAt: ev.startsAt,
        endsAt: ev.endsAt,
        cover: ev.cover,
        images: ev.images,
        maxAttendees: ev.maxAttendees,
        categories: ev.categories,
        currency: "eur",
        createdBy: adminUserId,
      });
      console.log(`  ✓ ${ev.title}`);
      created++;
    } catch {
      console.log(`  ⚠️  Skipped: ${ev.title}`);
      skipped++;
    }
  }

  console.log(`  → ${created} events created, ${skipped} skipped`);
}
