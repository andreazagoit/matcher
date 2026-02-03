import "dotenv/config";
import postgres from "postgres";

const connectionString = process.env.DATABASE_URL;

if (!connectionString) {
  console.error("❌ DATABASE_URL environment variable is not set");
  console.error("   Please set DATABASE_URL in your .env file");
  process.exit(1);
}

const client = postgres(connectionString);

async function drop() {
  console.log("🗑️  Dropping all tables...");

  try {
    await client`DROP SCHEMA public CASCADE`;
    await client`CREATE SCHEMA public`;
    console.log("✅ All tables dropped!");
  } catch (error) {
    console.error("❌ Drop failed:", error);
    if (error instanceof Error) {
      if (error.message.includes("ECONNREFUSED")) {
        console.error("\n💡 Connection refused. Possible issues:");
        console.error("   - Database server is not running");
        console.error("   - DATABASE_URL is incorrect");
        console.error("   - Network/firewall blocking connection");
        console.error("   - Supabase project might be paused");
      }
    }
    process.exit(1);
  } finally {
    await client.end();
  }

  process.exit(0);
}

drop();
