import { createSpace } from "./operations";
import type { CreateSpaceInput } from "./validator";

type SeedSpace = Omit<CreateSpaceInput, "ownerId">;

const SEED_SPACES: SeedSpace[] = [
  {
    name: "Matcher System",
    slug: "matcher-system",
    description: "Official Matcher System Space for internal use",
    cover: "https://images.unsplash.com/photo-1518770660439-4636190af475?w=800&q=80",
    images: [],
    visibility: "hidden",
    joinPolicy: "invite_only",
    categories: [],
  },
  // sport
  {
    name: "Fitness Partners",
    slug: "fitness-partners",
    description: "Trova il tuo partner di allenamento... e forse di più",
    cover: "https://images.unsplash.com/photo-1534438327276-14e5300c3a48?w=800&q=80",
    images: [],
    visibility: "public",
    joinPolicy: "open",
    categories: ["sport", "wellness"],
  },
  // outdoor
  {
    name: "Outdoor Adventures",
    slug: "outdoor-adventures",
    description: "Per chi ama la montagna, il trekking e le avventure all'aria aperta",
    cover: "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=800&q=80",
    images: [
      "https://images.unsplash.com/photo-1501854140801-50d01698950b?w=800&q=80",
      "https://images.unsplash.com/photo-1486870591958-9b9d0d1dda99?w=800&q=80",
    ],
    visibility: "public",
    joinPolicy: "open",
    categories: ["outdoor", "sport", "photography"],
  },
  // music
  {
    name: "Music & Live",
    slug: "music-live",
    description: "Concerti, jam session e amanti della musica dal vivo",
    cover: "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=800&q=80",
    images: [],
    visibility: "public",
    joinPolicy: "open",
    categories: ["music", "nightlife", "social"],
  },
  // art
  {
    name: "Artists & Creatives",
    slug: "artists-creatives",
    description: "Dove le menti creative si incontrano",
    cover: "https://images.unsplash.com/photo-1513364776144-60967b0f800f?w=800&q=80",
    images: [
      "https://images.unsplash.com/photo-1460661419201-fd4cecdf8a8b?w=800&q=80",
      "https://images.unsplash.com/photo-1547891654-e66ed7ebb968?w=800&q=80",
    ],
    visibility: "public",
    joinPolicy: "open",
    categories: ["art", "photography", "crafts"],
  },
  // food
  {
    name: "Foodies & Wine",
    slug: "foodies-wine",
    description: "Buon cibo, buon vino, buona compagnia",
    cover: "https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=800&q=80",
    images: [
      "https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=800&q=80",
      "https://images.unsplash.com/photo-1504754524776-8f4f37790ca0?w=800&q=80",
    ],
    visibility: "public",
    joinPolicy: "open",
    categories: ["food", "social", "nightlife"],
  },
  // social / dating
  {
    name: "Milano Singles",
    slug: "milano-singles",
    description: "Community per single milanesi alla ricerca dell'anima gemella",
    cover: "https://images.unsplash.com/photo-1517732306149-e8f829eb588a?w=800&q=80",
    images: [],
    visibility: "public",
    joinPolicy: "open",
    categories: ["travel", "social", "cinema"],
  },
  // wellness
  {
    name: "Mind & Body",
    slug: "mind-body",
    description: "Yoga, meditazione e benessere per incontrare persone consapevoli",
    cover: "https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?w=800&q=80",
    images: [
      "https://images.unsplash.com/photo-1506126613408-eca07ce68773?w=800&q=80",
      "https://images.unsplash.com/photo-1599901860904-17e6ed7083a0?w=800&q=80",
    ],
    visibility: "public",
    joinPolicy: "open",
    categories: ["wellness", "sport", "spirituality"],
  },
  // tech
  {
    name: "Tech Innovators",
    slug: "tech-innovators",
    description: "Una community per appassionati di tecnologia e innovazione",
    cover: "https://images.unsplash.com/photo-1518770660439-4636190af475?w=800&q=80",
    images: [
      "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b?w=800&q=80",
      "https://images.unsplash.com/photo-1461749280684-dccba630e2f6?w=800&q=80",
    ],
    visibility: "public",
    joinPolicy: "open",
    categories: ["tech", "entrepreneurship", "science"],
  },
  // culture
  {
    name: "Book & Culture Club",
    slug: "book-culture-club",
    description: "Lettori, cinefili e amanti della cultura si incontrano qui",
    cover: "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=800&q=80",
    images: [],
    visibility: "public",
    joinPolicy: "open",
    categories: ["culture", "cinema", "art"],
  },
  // cinema
  {
    name: "Cinema Lovers",
    slug: "cinema-lovers",
    description: "Film, serie TV e discussioni post-proiezione",
    cover: "https://images.unsplash.com/photo-1489599849927-2ee91cede3ba?w=800&q=80",
    images: [],
    visibility: "public",
    joinPolicy: "open",
    categories: ["cinema", "culture", "social"],
  },
  // social / nightlife
  {
    name: "Roma Dating",
    slug: "roma-dating",
    description: "Incontri romantici nella città eterna",
    cover: "https://images.unsplash.com/photo-1555992336-03a23c7b20ee?w=800&q=80",
    images: [],
    visibility: "public",
    joinPolicy: "open",
    categories: ["social", "nightlife", "food", "travel"],
  },
  // dance
  {
    name: "Dance & Move",
    slug: "dance-move",
    description: "Salsa, tango, hip-hop: balla e connettiti",
    cover: "https://images.unsplash.com/photo-1504609773096-104ff2c73ba4?w=800&q=80",
    images: [],
    visibility: "public",
    joinPolicy: "open",
    categories: ["dance", "music", "social"],
  },
  // animals
  {
    name: "Pet Lovers",
    slug: "pet-lovers",
    description: "Incontra altri amanti degli animali nei parchi cittadini",
    cover: "https://images.unsplash.com/photo-1450778869180-41d0601e046e?w=800&q=80",
    images: [],
    visibility: "public",
    joinPolicy: "open",
    categories: ["animals", "outdoor", "social"],
  },
  // sustainability
  {
    name: "Green Community",
    slug: "green-community",
    description: "Stile di vita sostenibile, volontariato ambientale e natura",
    cover: "https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?w=800&q=80",
    images: [],
    visibility: "public",
    joinPolicy: "open",
    categories: ["sustainability", "volunteering", "outdoor"],
  },
  // languages
  {
    name: "Language Exchange",
    slug: "language-exchange",
    description: "Parla, impara, connettiti — scambi linguistici dal vivo",
    cover: "https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8?w=800&q=80",
    images: [],
    visibility: "public",
    joinPolicy: "open",
    categories: ["languages", "culture", "social"],
  },
  // comedy
  {
    name: "Comedy & Fun",
    slug: "comedy-fun",
    description: "Stand-up, impro e serate di cabaret",
    cover: "https://images.unsplash.com/photo-1527224538127-2104bb71c51b?w=800&q=80",
    images: [],
    visibility: "public",
    joinPolicy: "open",
    categories: ["comedy", "social", "nightlife"],
  },
  // fashion
  {
    name: "Style & Fashion",
    slug: "style-fashion",
    description: "Moda, stile e incontri nel mondo del fashion",
    cover: "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=800&q=80",
    images: [],
    visibility: "public",
    joinPolicy: "open",
    categories: ["fashion", "art", "social"],
  },
  // entrepreneurship / science
  {
    name: "Tech Professionals",
    slug: "tech-professionals",
    description: "Professionisti del tech e imprenditori digitali",
    cover: "https://images.unsplash.com/photo-1497366216548-37526070297c?w=800&q=80",
    images: [],
    visibility: "public",
    joinPolicy: "apply",
    categories: ["entrepreneurship", "tech", "science"],
  },
  // photography / crafts
  {
    name: "Creators Hub",
    slug: "creators-hub",
    description: "Fotografi, artigiani e maker si ritrovano qui",
    cover: "https://images.unsplash.com/photo-1452802447250-470a88ac82bc?w=800&q=80",
    images: [],
    visibility: "public",
    joinPolicy: "open",
    categories: ["photography", "crafts", "art"],
  },
  // volunteering / spirituality
  {
    name: "Soul & Service",
    slug: "soul-service",
    description: "Volontariato, meditazione e crescita personale",
    cover: "https://images.unsplash.com/photo-1469571486292-0ba58a3f068b?w=800&q=80",
    images: [],
    visibility: "public",
    joinPolicy: "open",
    categories: ["volunteering", "spirituality", "wellness"],
  },
];

export async function seedSpaces(adminUserId: string) {
  console.log(`\n🏠 Seeding ${SEED_SPACES.length} spaces...`);

  for (const space of SEED_SPACES) {
    try {
      const { space: created } = await createSpace({
        ...space,
        ownerId: adminUserId,
      });
      console.log(`  ✓ ${created.name} (${created.slug})`);
    } catch {
      console.log(`  ⚠️  Skipped: ${space.name} (already exists)`);
    }
  }

  console.log(`  → spaces seeded`);
}
