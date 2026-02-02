import postgres from "postgres";

const client = postgres(process.env.DATABASE_URL!);

async function drop() {
  console.log("🗑️  Dropping all tables...");

  try {
    await client`DROP SCHEMA public CASCADE`;
    await client`CREATE SCHEMA public`;
    console.log("✅ All tables dropped!");
  } catch (error) {
    console.error("❌ Drop failed:", error);
    process.exit(1);
  }

  await client.end();
  process.exit(0);
}

drop();
