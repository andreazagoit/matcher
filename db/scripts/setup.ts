import "dotenv/config";
import postgres from "postgres";

const client = postgres(process.env.DATABASE_URL!);

async function setup() {
  console.log("📇 Creating vector indexes...\n");

  try {
    // Crea indici HNSW per vector search
    await client`
      CREATE INDEX IF NOT EXISTS values_embedding_idx 
      ON users 
      USING hnsw (values_embedding vector_cosine_ops)
    `;
    console.log("✅ Created values_embedding_idx");

    await client`
      CREATE INDEX IF NOT EXISTS interests_embedding_idx 
      ON users 
      USING hnsw (interests_embedding vector_cosine_ops)
    `;
    console.log("✅ Created interests_embedding_idx\n");

    console.log("✅ Vector indexes created!");
  } catch (error) {
    console.error("❌ Setup failed:", error);
    process.exit(1);
  }

  await client.end();
  process.exit(0);
}

setup();
