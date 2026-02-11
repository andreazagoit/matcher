import OpenAI from "openai";

/** Supported embedding axes */
export type EmbeddingAxis = "psychological" | "values" | "interests" | "behavioral";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

/**
 * Generates an embedding vector for a given descriptive text.
 * 
 * @param text - Semantic text (e.g., "Introverted persona, empathetic...")
 * @returns Embedding vector (1536 dimensions)
 */
export async function generateEmbedding(text: string): Promise<number[]> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is required");
  }

  if (!text || text.trim().length === 0) {
    throw new Error("Cannot generate embedding for empty text");
  }

  const response = await openai.embeddings.create({
    model: "text-embedding-3-small", // 1536 dimensions
    input: text.trim(),
  });

  return response.data[0].embedding;
}

/**
 * Generates embeddings in batch for multiple texts (performance optimized).
 */
export async function generateEmbeddingsBatch(
  texts: string[]
): Promise<number[][]> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is required");
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

// ============================================
// AXIS-SPECIFIC EMBEDDINGS (from assessment)
// ============================================

/** Embedding results for all descriptive axes */
export interface UserEmbeddings {
  psychological: number[] | null;
  values: number[] | null;
  interests: number[] | null;
  behavioral: number[] | null;
}

/**
 * Generates embeddings for all axes from their respective textual descriptions.
 * 
 * @param descriptions - Textual descriptions for each axis.
 * @returns Embeddings for each axis (null if metadata is missing).
 * 
 * @example
 * const descriptions = {
 *   psychological: "Introverted, empathetic, reflective...",
 *   values: "Priorities: family, personal growth...",
 *   interests: "Passions: trekking, reading, cinema...",
 *   behavioral: "Responds slowly, prefers deep conversations..."
 * };
 * 
 * const embeddings = await generateAllUserEmbeddings(descriptions);
 */
export async function generateAllUserEmbeddings(
  descriptions: Record<EmbeddingAxis, string>
): Promise<UserEmbeddings> {
  const axes: EmbeddingAxis[] = ["psychological", "values", "interests", "behavioral"];

  // Filtra assi con descrizioni valide
  const validAxes = axes.filter(
    (axis) => descriptions[axis] && descriptions[axis].trim().length > 0
  );

  if (validAxes.length === 0) {
    return {
      psychological: null,
      values: null,
      interests: null,
      behavioral: null,
    };
  }

  // Perform batch embedding generation for efficiency
  const textsToEmbed = validAxes.map((axis) => descriptions[axis]);
  const embeddings = await generateEmbeddingsBatch(textsToEmbed);

  // Mappa risultati
  const result: UserEmbeddings = {
    psychological: null,
    values: null,
    interests: null,
    behavioral: null,
  };

  validAxes.forEach((axis, index) => {
    result[axis] = embeddings[index];
  });

  return result;
}



