import "dotenv/config";
import { db } from "../drizzle";
import { users } from "../../models/users/schema";
import { embeddings } from "../../models/embeddings/schema";
import { eq, and } from "drizzle-orm";
import { sql } from "drizzle-orm";

const EMAIL = "marco.ferrari@example.com";

async function main() {
  const user = await db.query.users.findFirst({ where: eq(users.email, EMAIL) });
  if (!user) { console.log("❌ User not found"); process.exit(1); }
  console.log(`✅ User: ${user.name} (${user.id})`);
  console.log(`   gender=${user.gender} smoking=${user.smoking} drinking=${user.drinking} activityLevel=${user.activityLevel}`);

  const userEmb = await db.query.embeddings.findFirst({
    where: and(eq(embeddings.entityId, user.id), eq(embeddings.entityType, "user")),
    columns: { id: true, updatedAt: true },
  });
  console.log(`\n📦 User embedding: ${userEmb ? `✅ exists (updated ${userEmb.updatedAt})` : "❌ MISSING"}`);

  const counts = await db.execute<{ entity_type: string; count: string }>(sql`
    SELECT entity_type, COUNT(*) as count FROM embeddings GROUP BY entity_type ORDER BY entity_type
  `);
  console.log("\n📊 Embeddings per tipo:");
  for (const r of counts) console.log(`   ${r.entity_type}: ${r.count}`);

  // Check events
  const eventEmbs = await db.execute<{ count: string }>(sql`
    SELECT COUNT(*) as count FROM embeddings WHERE entity_type = 'event'
  `);
  console.log(`\n🎟  Events con embedding: ${eventEmbs[0]?.count ?? 0}`);

  // Check categories
  const catEmbs = await db.execute<{ count: string }>(sql`
    SELECT COUNT(*) as count FROM embeddings WHERE entity_type = 'category'
  `);
  console.log(`🏷️  Categories con embedding: ${catEmbs[0]?.count ?? 0}`);

  // If user has embedding, try similarity query
  if (userEmb) {
    const embRow = await db.query.embeddings.findFirst({
      where: and(eq(embeddings.entityId, user.id), eq(embeddings.entityType, "user")),
      columns: { embedding: true },
    });
    if (embRow) {
      const vec = `[${embRow.embedding.join(",")}]`;
      const nearEvents = await db.execute<{ entity_id: string; dist: number }>(sql`
        SELECT entity_id, embedding <=> ${sql.raw(`'${vec}'::vector`)} as dist
        FROM embeddings WHERE entity_type = 'event'
        ORDER BY dist LIMIT 3
      `);
      console.log(`\n🔍 Top 3 eventi più vicini per cosine similarity:`);
      for (const r of nearEvents) console.log(`   event ${r.entity_id}  dist=${Number(r.dist).toFixed(4)}`);

      const nearCats = await db.execute<{ entity_id: string; dist: number }>(sql`
        SELECT entity_id, embedding <=> ${sql.raw(`'${vec}'::vector`)} as dist
        FROM embeddings WHERE entity_type = 'category'
        ORDER BY dist LIMIT 4
      `);
      console.log(`\n🔍 Top 4 categorie più vicine:`);
      for (const r of nearCats) console.log(`   ${r.entity_id}  dist=${Number(r.dist).toFixed(4)}`);
    }
  }

  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
