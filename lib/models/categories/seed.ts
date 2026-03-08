import { CATEGORIES } from "./data";
import { createCategory } from "./operations";

export async function seedCategories() {
  console.log(`\n🏷️  Seeding ${CATEGORIES.length} categories...`);
  let created = 0;

  for (const name of CATEGORIES) {
    await createCategory(name);
    console.log(`  ✓ ${name}`);
    created++;
  }

  console.log(`  → ${created} categories seeded`);
}
