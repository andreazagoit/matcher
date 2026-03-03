/**
 * ML Service Client
 *
 * Typed wrappers for the Python FastAPI embedding server (default: http://localhost:8000).
 * Each function calls the corresponding /embed/* endpoint and returns the 256-dim
 * embedding vector to be stored in the `embeddings` table.
 *
 * If the service is unavailable the functions return null — callers decide
 * whether to silently skip or retry.
 */

const ML_SERVICE_URL =
  process.env.ML_SERVICE_URL ?? "http://localhost:8000";

// ── Shared helpers ────────────────────────────────────────────────────────────

async function postEmbed<T>(path: string, body: unknown): Promise<T | null> {
  try {
    const res = await fetch(`${ML_SERVICE_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      console.warn(`[ml] ${path} returned ${res.status}`);
      return null;
    }
    return res.json() as Promise<T>;
  } catch (err) {
    console.warn(`[ml] ${path} unreachable:`, err);
    return null;
  }
}

// ── Per-entity embedding generators ──────────────────────────────────────────

/**
 * Generate a 256-dim embedding for a user profile.
 * Returns null if the ML service is unavailable.
 */
export async function embedUser(params: {
  birthdate?: string | null;
  gender?: string | null;
  relationshipIntent?: string[];
  smoking?: string | null;
  drinking?: string | null;
  activityLevel?: string | null;
}): Promise<number[] | null> {
  const data = await postEmbed<{ embedding: number[] }>("/embed/user", {
    birthdate: params.birthdate ?? null,
    gender: params.gender ?? null,
    relationshipIntent: params.relationshipIntent ?? [],
    smoking: params.smoking ?? null,
    drinking: params.drinking ?? null,
    activityLevel: params.activityLevel ?? null,
  });
  return data?.embedding ?? null;
}

/**
 * Generate a 256-dim embedding for an event.
 * Returns null if the ML service is unavailable.
 */
export async function embedEvent(params: {
  categories?: string[];
  startsAt?: Date | string | null;
  attendeeCount?: number;
  maxAttendees?: number | null;
  isPaid?: boolean;
  priceCents?: number | null;
}): Promise<number[] | null> {
  const startsAt =
    params.startsAt instanceof Date
      ? params.startsAt.toISOString()
      : (params.startsAt ?? null);

  const now = new Date();
  const daysUntilEvent =
    params.startsAt
      ? Math.round(
          (new Date(params.startsAt).getTime() - now.getTime()) /
            (1000 * 60 * 60 * 24),
        )
      : null;

  const data = await postEmbed<{ embedding: number[] }>("/embed/event", {
    categories: params.categories ?? [],
    startsAt,
    attendeeCount: params.attendeeCount ?? 0,
    daysUntilEvent,
    maxAttendees: params.maxAttendees ?? null,
    isPaid: params.isPaid ?? false,
    priceCents: params.priceCents ?? null,
  });
  return data?.embedding ?? null;
}

/**
 * Generate a 256-dim embedding for a space.
 * Returns null if the ML service is unavailable.
 */
export async function embedSpace(params: {
  categories?: string[];
  memberCount?: number;
  eventCount?: number;
}): Promise<number[] | null> {
  const data = await postEmbed<{ embedding: number[] }>("/embed/space", {
    categories: params.categories ?? [],
    memberCount: params.memberCount ?? 0,
    eventCount: params.eventCount ?? 0,
  });
  return data?.embedding ?? null;
}

/**
 * Generate embeddings for a category.
 * Returns both the 256-dim graph embedding (for `embeddings` table)
 * and the 64-dim semantic embedding (for `categories.embedding`).
 * Returns null if the ML service is unavailable.
 */
export async function embedCategory(name: string): Promise<{
  embedding: number[];       // 256-dim — embeddings table
  categoryEmbedding: number[]; // 64-dim  — categories.embedding
  modelUsed: string;
} | null> {
  const data = await postEmbed<{
    embedding: number[];
    category_embedding: number[];
    model_used: string;
  }>("/embed/category", { name });

  if (!data) return null;
  return {
    embedding: data.embedding,
    categoryEmbedding: data.category_embedding,
    modelUsed: data.model_used,
  };
}
