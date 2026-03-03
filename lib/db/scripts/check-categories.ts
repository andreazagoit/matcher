import "dotenv/config";
import { db } from "../drizzle";
import { sql } from "drizzle-orm";

async function main() {
  // Categories vs events match
  const cats = await db.execute<{ id: string }>(sql`SELECT id FROM categories ORDER BY id`);
  
  console.log("\n📊 Categoria → eventi con quel tag:");
  for (const cat of cats) {
    const count = await db.execute<{ count: string }>(sql`
      SELECT COUNT(*) as count FROM events 
      WHERE categories && ARRAY[${cat.id}]::text[]
        AND starts_at >= NOW()
    `);
    const n = Number(count[0]?.count ?? 0);
    console.log(`  ${n > 0 ? "✅" : "❌"} ${cat.id.padEnd(20)} ${n} eventi futuri`);
  }

  // What tags do events actually use?
  const eventTags = await db.execute<{ categories: string[] }>(sql`
    SELECT DISTINCT unnest(categories) as categories FROM events ORDER BY 1
  `);
  console.log("\n🏷️  Tag usati negli eventi:");
  console.log(" ", eventTags.map(r => r.categories).join(", "));

  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
