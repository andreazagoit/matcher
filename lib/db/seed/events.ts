import { db } from "../drizzle";
import { events } from "../../models/events/schema";
import { spaces } from "../../models/spaces/schema";

interface SeedEvent {
  spaceSlug: string;
  title: string;
  description: string;
  location: string;
  lat: number;
  lon: number;
  startsAt: Date;
  endsAt?: Date;
  maxAttendees?: number;
  tags: string[];
}

const now = new Date();
const d = (daysFromNow: number, hour = 18, minute = 0) => {
  const date = new Date(now);
  date.setDate(date.getDate() + daysFromNow);
  date.setHours(hour, minute, 0, 0);
  return date;
};

const SEED_EVENTS: SeedEvent[] = [
  // Milano Singles
  {
    spaceSlug: "milano-singles",
    title: "Aperitivo al Naviglio Grande",
    description: "Aperitivo di gruppo sui Navigli, ottimo per conoscere nuove persone in un'atmosfera rilassata.",
    location: "Naviglio Grande, Milano",
    lat: 45.4545,
    lon: 9.1718,
    startsAt: d(3, 19, 0),
    endsAt: d(3, 22, 0),
    maxAttendees: 20,
    tags: ["parties", "restaurants"],
  },
  {
    spaceSlug: "milano-singles",
    title: "Speed Dating al Brera",
    description: "Classico speed dating nel quartiere Brera. 5 minuti per fare colpo!",
    location: "Caffè Letterario Brera, Milano",
    lat: 45.4722,
    lon: 9.1864,
    startsAt: d(7, 20, 0),
    endsAt: d(7, 23, 0),
    maxAttendees: 30,
    tags: ["parties", "cinema"],
  },
  {
    spaceSlug: "milano-singles",
    title: "Serata Cinema Anteo",
    description: "Film + discussione post-proiezione. Un modo diverso per conoscersi.",
    location: "Anteo Palazzo del Cinema, Milano",
    lat: 45.4829,
    lon: 9.1786,
    startsAt: d(12, 20, 30),
    endsAt: d(12, 23, 30),
    maxAttendees: 25,
    tags: ["cinema"],
  },

  // Roma Dating
  {
    spaceSlug: "roma-dating",
    title: "Cena a Trastevere",
    description: "Cena romantica in gruppo nel cuore di Trastevere. Prenotiamo un tavolo per 12.",
    location: "Piazza di Santa Maria in Trastevere, Roma",
    lat: 41.8897,
    lon: 12.4699,
    startsAt: d(4, 20, 0),
    endsAt: d(4, 23, 30),
    maxAttendees: 12,
    tags: ["restaurants", "wine"],
  },
  {
    spaceSlug: "roma-dating",
    title: "Tour al Colosseo al tramonto",
    description: "Visita guidata al Colosseo durante il tramonto. Atmosfera unica.",
    location: "Colosseo, Roma",
    lat: 41.8902,
    lon: 12.4922,
    startsAt: d(9, 17, 30),
    endsAt: d(9, 20, 0),
    maxAttendees: 15,
    tags: ["museums", "travel"],
  },

  // Outdoor Adventures
  {
    spaceSlug: "outdoor-adventures",
    title: "Trekking al Monte Generoso",
    description: "Escursione di media difficoltà con vista panoramica su Lombardia e Svizzera.",
    location: "Monte Generoso, Lombardia",
    lat: 45.9319,
    lon: 9.0197,
    startsAt: d(5, 8, 0),
    endsAt: d(5, 17, 0),
    maxAttendees: 15,
    tags: ["trekking", "mountains"],
  },
  {
    spaceSlug: "outdoor-adventures",
    title: "Arrampicata alle Dolomiti",
    description: "Weekend di arrampicata per tutti i livelli nelle Dolomiti Bellunesi.",
    location: "Cortina d'Ampezzo, Belluno",
    lat: 46.5362,
    lon: 12.1355,
    startsAt: d(14, 7, 0),
    endsAt: d(16, 19, 0),
    maxAttendees: 10,
    tags: ["climbing", "mountains", "camping"],
  },
  {
    spaceSlug: "outdoor-adventures",
    title: "Ciclismo sul Lago di Garda",
    description: "Giro in bici lungo la sponda bresciana del Lago di Garda. ~40km, livello medio.",
    location: "Desenzano del Garda, Brescia",
    lat: 45.4664,
    lon: 10.5351,
    startsAt: d(8, 9, 0),
    endsAt: d(8, 14, 0),
    maxAttendees: 20,
    tags: ["cycling"],
  },

  // Book Lovers Club
  {
    spaceSlug: "book-lovers-club",
    title: "Book Club: Il nome della rosa",
    description: "Discussione del capolavoro di Umberto Eco. Portate le vostre copie annotate!",
    location: "Libreria Feltrinelli, Piazza Piemonte 2, Milano",
    lat: 45.4721,
    lon: 9.1651,
    startsAt: d(6, 18, 30),
    endsAt: d(6, 21, 0),
    maxAttendees: 15,
    tags: ["reading", "writing"],
  },
  {
    spaceSlug: "book-lovers-club",
    title: "Laboratorio di Scrittura Creativa",
    description: "Workshop pratico di scrittura creativa con un autore ospite.",
    location: "Centro Culturale di Milano",
    lat: 45.4641,
    lon: 9.1919,
    startsAt: d(18, 10, 0),
    endsAt: d(18, 13, 0),
    maxAttendees: 12,
    tags: ["writing", "art"],
  },

  // Foodies & Wine
  {
    spaceSlug: "foodies-wine",
    title: "Degustazione Vini della Toscana",
    description: "Serata di degustazione con sommelier professionista. 6 etichette selezionate.",
    location: "Enoteca Pinchiorri, Firenze",
    lat: 43.7711,
    lon: 11.2600,
    startsAt: d(10, 19, 30),
    endsAt: d(10, 22, 30),
    maxAttendees: 16,
    tags: ["wine", "restaurants"],
  },
  {
    spaceSlug: "foodies-wine",
    title: "Cooking Class: Pasta Fresca",
    description: "Impariamo a fare la pasta fresca all'uovo con uno chef professionista.",
    location: "Scuola di Cucina Eataly, Milano",
    lat: 45.4641,
    lon: 9.2026,
    startsAt: d(13, 11, 0),
    endsAt: d(13, 14, 30),
    maxAttendees: 12,
    tags: ["cooking"],
  },

  // Fitness Partners
  {
    spaceSlug: "fitness-partners",
    title: "Running in Parco Sempione",
    description: "Corsa di gruppo al mattino nel parco. Tutti i ritmi benvenuti!",
    location: "Parco Sempione, Milano",
    lat: 45.4754,
    lon: 9.1742,
    startsAt: d(2, 7, 30),
    endsAt: d(2, 9, 0),
    maxAttendees: 25,
    tags: ["running"],
  },
  {
    spaceSlug: "fitness-partners",
    title: "Yoga all'alba a Villa Borghese",
    description: "Sessione di yoga all'aperto al sorgere del sole. Porta il tuo tappetino.",
    location: "Villa Borghese, Roma",
    lat: 41.9136,
    lon: 12.4921,
    startsAt: d(4, 6, 30),
    endsAt: d(4, 8, 0),
    maxAttendees: 20,
    tags: ["yoga"],
  },

  // Tech Innovators
  {
    spaceSlug: "tech-innovators",
    title: "Hackathon AI Weekend",
    description: "48 ore per costruire un progetto AI. Team di 3-4 persone. Premio per il migliore!",
    location: "Impact Hub, Milano",
    lat: 45.4669,
    lon: 9.1890,
    startsAt: d(20, 9, 0),
    endsAt: d(22, 17, 0),
    maxAttendees: 40,
    tags: ["coding", "gaming"],
  },
  {
    spaceSlug: "tech-innovators",
    title: "Meetup: LLM in produzione",
    description: "Talk tecnico su come portare modelli linguistici in produzione. Q&A aperto.",
    location: "Google for Startups Campus, Milano",
    lat: 45.4654,
    lon: 9.1856,
    startsAt: d(11, 18, 30),
    endsAt: d(11, 21, 0),
    maxAttendees: 60,
    tags: ["coding"],
  },

  // Artists & Creatives
  {
    spaceSlug: "artists-creatives",
    title: "Open Studio a Isola",
    description: "Visita agli atelier degli artisti del quartiere Isola. Apertura straordinaria.",
    location: "Quartiere Isola, Milano",
    lat: 45.4890,
    lon: 9.1908,
    startsAt: d(6, 15, 0),
    endsAt: d(6, 20, 0),
    maxAttendees: 30,
    tags: ["art", "photography"],
  },
  {
    spaceSlug: "artists-creatives",
    title: "Concerto Jazz al Blue Note",
    description: "Serata jazz con musicisti emergenti. Cena opzionale inclusa.",
    location: "Blue Note Milano",
    lat: 45.4891,
    lon: 9.1973,
    startsAt: d(15, 21, 0),
    endsAt: d(15, 24, 0),
    maxAttendees: 50,
    tags: ["music"],
  },
];

export async function seedEvents(adminUserId: string) {
  console.log(`\n📅 Seeding ${SEED_EVENTS.length} events...`);

  // Load all spaces once
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
      await db.insert(events).values({
        spaceId,
        title: ev.title,
        description: ev.description,
        location: ev.location,
        coordinates: { x: ev.lon, y: ev.lat },
        startsAt: ev.startsAt,
        endsAt: ev.endsAt,
        maxAttendees: ev.maxAttendees,
        status: "published",
        tags: ev.tags,
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
