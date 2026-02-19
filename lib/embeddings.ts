import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

/**
 * Generate an embedding vector for a given text.
 * Uses OpenAI text-embedding-3-small (1536 dimensions).
 */
export async function generateEmbedding(text: string): Promise<number[]> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is required for embeddings");
  }

  if (!text || text.trim().length === 0) {
    throw new Error("Cannot generate embedding for empty text");
  }

  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text.trim(),
  });

  return response.data[0].embedding;
}

/**
 * Generate embeddings in batch for multiple texts.
 */
export async function generateEmbeddingsBatch(
  texts: string[],
): Promise<number[][]> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is required for embeddings");
  }

  const validTexts = texts.filter((t) => t && t.trim().length > 0);
  if (validTexts.length === 0) {
    throw new Error("No valid texts to embed");
  }

  const response = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: validTexts.map((t) => t.trim()),
  });

  return response.data.map((d) => d.embedding);
}

/**
 * Compute the centroid (average) of multiple embedding vectors.
 */
export function computeCentroid(vectors: number[][]): number[] {
  if (vectors.length === 0) return [];
  const dims = vectors[0].length;
  const centroid = new Array(dims).fill(0);

  for (const vec of vectors) {
    for (let i = 0; i < dims; i++) {
      centroid[i] += vec[i];
    }
  }

  for (let i = 0; i < dims; i++) {
    centroid[i] /= vectors.length;
  }

  return centroid;
}
