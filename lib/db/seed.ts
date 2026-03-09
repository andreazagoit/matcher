import "dotenv/config";
import { seedCategories } from "../models/categories/seed";
import { seedUsers } from "../models/users/seed";
import { seedSpaces } from "../models/spaces/seed";
import { seedEvents } from "../models/events/seed";
import { seedProfiles } from "../models/impressions/seed";

async function seed() {
  console.log("🌱 Starting seed...\n");

  try {
    await seedCategories();

    const usersByEmail = await seedUsers();

    const adminUser = usersByEmail["admin@matcher.local"];
    if (!adminUser) throw new Error("Admin user was not created");

    await seedSpaces(adminUser.id);
    await seedEvents(adminUser.id);

    const nonAdminIds = Object.entries(usersByEmail)
      .filter(([email]) => email !== "admin@matcher.local")
      .map(([, user]) => user.id);

    await seedProfiles(nonAdminIds);

    console.log("\n✅ Seed completed successfully!");
  } catch (error) {
    console.error("\n❌ Seed failed:", error);
    process.exit(1);
  }

  process.exit(0);
}

seed();
