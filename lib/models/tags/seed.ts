import "dotenv/config";
import { db } from "@/lib/db/drizzle";
import { tags } from "./schema";
import { TAG_CATEGORIES } from "./data";
import { createTag } from "./operations";

async function seedTags() {
    console.log("Seeding tags to Database delegating computation to Python ML (64d + 256d HGT)...");
    let count = 0;

    for (const [category, tagList] of Object.entries(TAG_CATEGORIES)) {
        for (const tagName of tagList) {
            try {
                // Delegate to operation which handles both ML API call + DB inserts
                await createTag(tagName, category);
                console.log(`[+] Seeded semantic tag: ${tagName} (${category})`);
                count++;
            } catch (err) {
                console.warn(`[!] Failed to seed tag: ${tagName}`, err);
            }
        }
    }

    console.log(`\n✅ Migration complete! Generated embeddings and inserted ${count} NEW tags.`);
}

seedTags()
    .catch((e) => {
        console.error("Migration failed:", e);
        process.exit(1);
    })
    .finally(() => {
        process.exit(0);
    });
