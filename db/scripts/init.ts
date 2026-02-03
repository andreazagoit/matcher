import "dotenv/config";
import postgres from "postgres";

const client = postgres(process.env.DATABASE_URL!);

async function init() {
  console.log("🔧 Initializing database extensions...");

  try {
    await client`CREATE EXTENSION IF NOT EXISTS vector`;
    console.log("✅ pgvector extension enabled!");
  } catch (error) {
    console.error("❌ Init failed:", error);
    process.exit(1);
  }

  await client.end();
  process.exit(0);
}

init();

