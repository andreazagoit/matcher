import "dotenv/config";
import { db } from "../drizzle";
import { users } from "../../models/users/schema";
import { createSpace } from "../../models/spaces/operations";
import { eq } from "drizzle-orm";

/**
 * Seed - Add 10 test spaces
 */

const TEST_SPACES = [
    {
        name: "Milano Singles",
        slug: "milano-singles",
        description: "Community per single milanesi alla ricerca dell'anima gemella",
        visibility: "public" as const,
        joinPolicy: "open" as const,
    },
    {
        name: "Roma Dating",
        slug: "roma-dating",
        description: "Incontri romantici nella città eterna",
        visibility: "public" as const,
        joinPolicy: "open" as const,
    },
    {
        name: "Outdoor Adventures",
        slug: "outdoor-adventures",
        description: "Per chi ama la montagna, il trekking e le avventure all'aria aperta",
        visibility: "public" as const,
        joinPolicy: "open" as const,
    },
    {
        name: "Book Lovers Club",
        slug: "book-lovers-club",
        description: "Connessioni profonde tra amanti della lettura",
        visibility: "public" as const,
        joinPolicy: "open" as const,
    },
    {
        name: "Foodies & Wine",
        slug: "foodies-wine",
        description: "Buon cibo, buon vino, buona compagnia",
        visibility: "public" as const,
        joinPolicy: "open" as const,
    },
    {
        name: "Tech Professionals",
        slug: "tech-professionals",
        description: "Dating per professionisti del settore tech",
        visibility: "public" as const,
        joinPolicy: "apply" as const,
    },
    {
        name: "Artists & Creatives",
        slug: "artists-creatives",
        description: "Dove le menti creative si incontrano",
        visibility: "public" as const,
        joinPolicy: "open" as const,
    },
    {
        name: "Fitness Partners",
        slug: "fitness-partners",
        description: "Trova il tuo partner di allenamento... e forse di più",
        visibility: "public" as const,
        joinPolicy: "open" as const,
    },
    {
        name: "30+ Connection",
        slug: "30-plus-connection",
        description: "Dating maturo per over 30",
        visibility: "public" as const,
        joinPolicy: "open" as const,
    },
    {
        name: "VIP Exclusive",
        slug: "vip-exclusive",
        description: "Community esclusiva su invito",
        visibility: "private" as const,
        joinPolicy: "invite_only" as const,
    },
];

async function seedSpaces() {
    console.log(`🌱 Adding ${TEST_SPACES.length} test spaces...\n`);

    try {
        // Find admin user to be the owner
        const adminUser = await db.query.users.findFirst({
            where: eq(users.email, "admin@matcher.local"),
        });

        if (!adminUser) {
            console.error("❌ Admin user not found. Run the main seed first.");
            process.exit(1);
        }

        for (const spaceData of TEST_SPACES) {
            try {
                const result = await createSpace({
                    ...spaceData,
                    ownerId: adminUser.id,
                });
                console.log(`  ✓ Created: ${result.space.name} (${result.space.slug})`);
            } catch (error) {
                // Likely already exists (unique slug constraint)
                console.log(`  ⚠️ Skipped: ${spaceData.name} (already exists)`);
            }
        }

        console.log(`\n✅ Spaces seeded successfully!`);
    } catch (error) {
        console.error("❌ Seed failed:", error);
        process.exit(1);
    }

    process.exit(0);
}

seedSpaces();
