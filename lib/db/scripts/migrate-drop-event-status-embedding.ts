import "dotenv/config";
import postgres from "postgres";

async function main() {
  if (!process.env.DATABASE_URL) {
    throw new Error("DATABASE_URL is required");
  }

  const sql = postgres(process.env.DATABASE_URL);
  try {
    console.log("Dropping events.status/events.embedding and event_status enum...");

    await sql.begin(async (tx) => {
      await tx`ALTER TABLE events DROP COLUMN IF EXISTS status`;
      await tx`ALTER TABLE events DROP COLUMN IF EXISTS embedding`;
      await tx`DROP TYPE IF EXISTS event_status`;
    });

    console.log("Migration completed.");
  } finally {
    await sql.end();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
