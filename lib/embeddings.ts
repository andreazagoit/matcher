import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

/**
 * Genera embedding per un array di valori/interessi
 */
export async function generateEmbedding(
  items: string[]
): Promise<number[]> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is required");
  }

  if (items.length === 0) {
    throw new Error("Cannot generate embedding for empty array");
  }

  // Crea un testo descrittivo dai valori/interessi
  const text = items.join(", ");

  const response = await openai.embeddings.create({
    model: "text-embedding-3-small", // 1536 dimensions
    input: text,
  });

  return response.data[0].embedding;
}

/**
 * Genera embeddings per values e interests
 */
export async function generateUserEmbeddings(
  values: string[],
  interests: string[]
): Promise<{ valuesEmbedding: number[]; interestsEmbedding: number[] }> {
  if (values.length === 0 || interests.length === 0) {
    throw new Error("Values and interests are required");
  }

  const [valuesEmbedding, interestsEmbedding] = await Promise.all([
    generateEmbedding(values),
    generateEmbedding(interests),
  ]);

  return { valuesEmbedding, interestsEmbedding };
}

