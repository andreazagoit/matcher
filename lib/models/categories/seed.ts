import "dotenv/config";
import { CATEGORIES } from "./data";
import { createCategory } from "./operations";

async function seedCategories() {
  console.log("Seeding categories (64d OpenAI + 256d HGT embeddings)...");
  let count = 0;
  for (const name of CATEGORIES) {
    try {
      await createCategory(name);
      console.log(`[+] ${name}`);
      count++;
    } catch (err) {
      console.warn(`[!] Failed: ${name}`, err);
    }
  }
  console.log(`\nDone — ${count} categories seeded.`);
}

seedCategories()
  .catch((e) => {
    console.error("Seed failed:", e);
    process.exit(1);
  })
  .finally(() => process.exit(0));
