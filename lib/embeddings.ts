import OpenAI from "openai";

/** Assi di embedding supportati */
export type EmbeddingAxis = "psychological" | "values" | "interests" | "behavioral";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY!,
});

/**
 * Genera embedding per un testo descrittivo
 * 
 * @param text - Testo semantico (es. "Persona introversa, empatica...")
 * @returns Vettore embedding (1536 dimensioni)
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
 * Genera embedding batch per più testi (più efficiente)
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
// EMBEDDINGS PER ASSI (dall'assessment)
// ============================================

/** Risultato embeddings per tutti gli assi */
export interface UserEmbeddings {
  psychological: number[] | null;
  values: number[] | null;
  interests: number[] | null;
  behavioral: number[] | null;
}

/**
 * Genera embeddings per tutti gli assi dalle descrizioni testuali
 * 
 * @param descriptions - Descrizioni testuali per ogni asse (output di transform.ts)
 * @returns Embeddings per ogni asse (null se descrizione mancante)
 * 
 * @example
 * const descriptions = {
 *   psychological: "Persona introversa, empatica, riflessiva...",
 *   values: "Priorità: famiglia, crescita personale...",
 *   interests: "Passioni: trekking, lettura, cinema...",
 *   behavioral: "Risponde lentamente, preferisce conversazioni profonde..."
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

  // Genera embeddings in batch (più efficiente)
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



