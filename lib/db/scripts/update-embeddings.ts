/**
 * Reads training-data/embeddings.json and upserts ML embeddings into the DB.
 *
 * ⚠️  Prerequisite: the `embeddings` table must use EMBEDDING_DIMENSIONS = 64.
 *     If currently set to 1536 (OpenAI), run the migration first:
 *
 *       ALTER TABLE embeddings DROP COLUMN embedding;
 *       ALTER TABLE embeddings ADD COLUMN embedding vector(64) NOT NULL;
 *       DROP INDEX IF EXISTS embeddings_hnsw_idx;
 *       CREATE INDEX embeddings_hnsw_idx ON embeddings
 *         USING hnsw (embedding vector_cosine_ops);
 *
 *     Then update lib/models/embeddings/schema.ts:
 *       export const EMBEDDING_DIMENSIONS = 64;
 *
 * Usage: npm run ml:update-embeddings
 */

import "dotenv/config";
import fs from "fs";
import path from "path";
import postgres from "postgres";

const client = postgres(process.env.DATABASE_URL!);

const EMBEDDINGS_FILE = path.resolve(
  process.cwd(),
  "python-ml/training-data/embeddings.json",
);

const BATCH_SIZE = 200; // rows per upsert batch

type EmbeddingRecord = {
  id: string;
  type: "user" | "event" | "space";
  embedding: number[];
};

async function upsertBatch(batch: EmbeddingRecord[]) {
  // Upsert in parallel within batch for speed
  await Promise.all(
    batch.map((r) =>
      client`
        INSERT INTO embeddings (entity_id, entity_type, embedding, updated_at)
        VALUES (
          ${r.id}::uuid,
          ${r.type},
          ${JSON.stringify(r.embedding)}::vector,
          NOW()
        )
        ON CONFLICT (entity_id, entity_type)
        DO UPDATE SET
          embedding  = EXCLUDED.embedding,
          updated_at = NOW()
      `,
    ),
  );
}

async function main() {
  if (!fs.existsSync(EMBEDDINGS_FILE)) {
    console.error(`File not found: ${EMBEDDINGS_FILE}`);
    console.error("Run 'npm run ml:reembed' first.");
    process.exit(1);
  }

  const records: EmbeddingRecord[] = JSON.parse(
    fs.readFileSync(EMBEDDINGS_FILE, "utf-8"),
  );

  const byType = records.reduce(
    (acc, r) => {
      acc[r.type] = (acc[r.type] ?? 0) + 1;
      return acc;
    },
    {} as Record<string, number>,
  );

  console.log(`Loaded ${records.length.toLocaleString()} embeddings:`);
  for (const [type, count] of Object.entries(byType)) {
    console.log(`  ${type.padEnd(8)} ${count.toLocaleString()}`);
  }
  console.log();

  // Check embedding dimension matches DB column
  const firstEmb = records[0]?.embedding;
  if (firstEmb) {
    const dim = firstEmb.length;
    console.log(`Embedding dimension: ${dim}  (DB column must match)`);
    if (dim !== 64) {
      console.warn(
        `⚠️  Expected 64-dim ML embeddings, got ${dim}. Check model config.`,
      );
    }
    console.log();
  }

  // Upsert in batches
  let done = 0;
  const total = records.length;
  const start = Date.now();

  for (let i = 0; i < records.length; i += BATCH_SIZE) {
    const batch = records.slice(i, i + BATCH_SIZE);
    await upsertBatch(batch);
    done += batch.length;
    const pct = ((done / total) * 100).toFixed(1);
    const rps = Math.round(done / ((Date.now() - start) / 1000));
    process.stdout.write(`\r  ${done.toLocaleString()}/${total.toLocaleString()}  (${pct}%  ~${rps} rows/s)`);
  }

  const elapsed = ((Date.now() - start) / 1000).toFixed(1);
  console.log(`\n\n✓ Upserted ${total.toLocaleString()} embeddings in ${elapsed}s`);

  await client.end();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
