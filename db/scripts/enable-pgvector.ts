import postgres from "postgres";

const client = postgres(process.env.DATABASE_URL!);

async function enablePgvector() {
  console.log("🔧 Enabling pgvector extension...");

  try {
    await client`CREATE EXTENSION IF NOT EXISTS vector`;
    console.log("✅ pgvector enabled!");
  } catch (error) {
    console.error("❌ Failed:", error);
    process.exit(1);
  }

  await client.end();
  process.exit(0);
}

enablePgvector();

