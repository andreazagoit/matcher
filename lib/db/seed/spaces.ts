import { createSpace } from "../../models/spaces/operations";

const SEED_SPACES = [
  {
    name: "Matcher System",
    slug: "matcher-system",
    description: "Official Matcher System Space for internal use",
    visibility: "hidden" as const,
    joinPolicy: "invite_only" as const,
  },
  {
    name: "Tech Innovators",
    slug: "tech-innovators",
    description: "A community for technology enthusiasts and innovators.",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    tags: ["coding", "gaming"],
  },
  {
    name: "Nature Lovers",
    slug: "nature-lovers",
    description: "Hiking, photography, and outdoor adventures.",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    tags: ["trekking", "camping", "mountains", "photography"],
  },
  {
    name: "Milano Singles",
    slug: "milano-singles",
    description: "Community per single milanesi alla ricerca dell'anima gemella",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    tags: ["travel", "parties", "restaurants", "cinema"],
  },
  {
    name: "Roma Dating",
    slug: "roma-dating",
    description: "Incontri romantici nella città eterna",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    tags: ["restaurants", "wine", "travel", "museums"],
  },
  {
    name: "Outdoor Adventures",
    slug: "outdoor-adventures",
    description: "Per chi ama la montagna, il trekking e le avventure all'aria aperta",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    tags: ["trekking", "climbing", "camping", "mountains", "cycling"],
  },
  {
    name: "Book Lovers Club",
    slug: "book-lovers-club",
    description: "Connessioni profonde tra amanti della lettura",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    tags: ["reading", "writing", "art", "theater"],
  },
  {
    name: "Foodies & Wine",
    slug: "foodies-wine",
    description: "Buon cibo, buon vino, buona compagnia",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    tags: ["cooking", "restaurants", "wine", "craft_beer", "street_food"],
  },
  {
    name: "Tech Professionals",
    slug: "tech-professionals",
    description: "Dating per professionisti del settore tech",
    visibility: "public" as const,
    joinPolicy: "apply" as const,
    tags: ["coding", "gaming", "coffee"],
  },
  {
    name: "Artists & Creatives",
    slug: "artists-creatives",
    description: "Dove le menti creative si incontrano",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    tags: ["art", "drawing", "music", "photography", "writing"],
  },
  {
    name: "Fitness Partners",
    slug: "fitness-partners",
    description: "Trova il tuo partner di allenamento... e forse di più",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    tags: ["gym", "running", "yoga", "swimming", "cycling"],
  },
  {
    name: "30+ Connection",
    slug: "30-plus-connection",
    description: "Dating maturo per over 30",
    visibility: "public" as const,
    joinPolicy: "open" as const,
    tags: ["wine", "restaurants", "travel", "cinema", "theater"],
  },
  {
    name: "VIP Exclusive",
    slug: "vip-exclusive",
    description: "Community esclusiva su invito",
    visibility: "private" as const,
    joinPolicy: "invite_only" as const,
    tags: ["wine", "travel", "art"],
  },
];

/**
 * Seed all spaces. Requires admin user ID as the owner.
 */
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
