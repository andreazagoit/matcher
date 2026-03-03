import { createSpace } from "../../models/spaces/operations";

const SEED_SPACES = [
  {
    name: "Matcher System",
    slug: "matcher-system",
    description: "Official Matcher System Space for internal use",
    visibility: "hidden" as const,
    joinPolicy: "invite_only" as const,
    categories: [],
  },
  // sport
  {
    name: "Fitness Partners",
    slug: "fitness-partners",
    description: "Trova il tuo partner di allenamento... e forse di più",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["sport", "wellness"],
  },
  // outdoor
  {
    name: "Outdoor Adventures",
    slug: "outdoor-adventures",
    description: "Per chi ama la montagna, il trekking e le avventure all'aria aperta",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["outdoor", "sport", "photography"],
  },
  // music
  {
    name: "Music & Live",
    slug: "music-live",
    description: "Concerti, jam session e amanti della musica dal vivo",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["music", "nightlife", "social"],
  },
  // art
  {
    name: "Artists & Creatives",
    slug: "artists-creatives",
    description: "Dove le menti creative si incontrano",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["art", "photography", "crafts"],
  },
  // food
  {
    name: "Foodies & Wine",
    slug: "foodies-wine",
    description: "Buon cibo, buon vino, buona compagnia",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["food", "social", "nightlife"],
  },
  // travel
  {
    name: "Milano Singles",
    slug: "milano-singles",
    description: "Community per single milanesi alla ricerca dell'anima gemella",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["travel", "social", "cinema"],
  },
  // wellness
  {
    name: "Mind & Body",
    slug: "mind-body",
    description: "Yoga, meditazione e benessere per incontrare persone consapevoli",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["wellness", "sport", "spirituality"],
  },
  // tech
  {
    name: "Tech Innovators",
    slug: "tech-innovators",
    description: "Una community per appassionati di tecnologia e innovazione",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["tech", "entrepreneurship", "science"],
  },
  // culture
  {
    name: "Book & Culture Club",
    slug: "book-culture-club",
    description: "Lettori, cinefili e amanti della cultura si incontrano qui",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["culture", "cinema", "art"],
  },
  // cinema
  {
    name: "Cinema Lovers",
    slug: "cinema-lovers",
    description: "Film, serie TV e discussioni post-proiezione",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["cinema", "culture", "social"],
  },
  // social / nightlife
  {
    name: "Roma Dating",
    slug: "roma-dating",
    description: "Incontri romantici nella città eterna",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["social", "nightlife", "food", "travel"],
  },
  // dance
  {
    name: "Dance & Move",
    slug: "dance-move",
    description: "Salsa, tango, hip-hop: balla e connettiti",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["dance", "music", "social"],
  },
  // animals
  {
    name: "Pet Lovers",
    slug: "pet-lovers",
    description: "Incontra altri amanti degli animali nei parchi cittadini",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["animals", "outdoor", "social"],
  },
  // sustainability
  {
    name: "Green Community",
    slug: "green-community",
    description: "Stile di vita sostenibile, volontariato ambientale e natura",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["sustainability", "volunteering", "outdoor"],
  },
  // languages
  {
    name: "Language Exchange",
    slug: "language-exchange",
    description: "Parla, impara, connettiti — scambi linguistici dal vivo",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["languages", "culture", "social"],
  },
  // comedy
  {
    name: "Comedy & Fun",
    slug: "comedy-fun",
    description: "Stand-up, impro e serate di cabaret",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["comedy", "social", "nightlife"],
  },
  // fashion
  {
    name: "Style & Fashion",
    slug: "style-fashion",
    description: "Moda, stile e incontri nel mondo del fashion",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["fashion", "art", "social"],
  },
  // entrepreneurship / science
  {
    name: "Tech Professionals",
    slug: "tech-professionals",
    description: "Professionisti del tech e imprenditori digitali",
    visibility: "public" as const,
    joinPolicy: "apply" as const,
    categories: ["entrepreneurship", "tech", "science"],
  },
  // photography / crafts
  {
    name: "Creators Hub",
    slug: "creators-hub",
    description: "Fotografi, artigiani e maker si ritrovano qui",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    categories: ["photography", "crafts", "art"],
  },
  // volunteering / spirituality
  {
    name: "Soul & Service",
    slug: "soul-service",
    description: "Volontariato, meditazione e crescita personale",
    visibility: "public" as const,
    joinPolicy: "open" as const,
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
