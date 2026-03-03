import "dotenv/config";
import { seedUsers } from "./users";
import { seedSpaces } from "./spaces";
import { seedCategories, seedProfiles } from "./interests";
import { seedEvents } from "./events";
import { seedProfileCards } from "./useritems";


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
    await seedProfileCards(nonAdminIds);


    console.log("\n✅ Seed completed successfully!");
  } catch (error) {
    console.error("\n❌ Seed failed:", error);
    process.exit(1);
  }

  process.exit(0);
}

seed();
