import { db } from "@/lib/db/drizzle";
import { impressions } from "./schema";

type ItemType = "user" | "event" | "space" | "category";
type Action = "shown" | "clicked" | "skipped" | "joined" | "messaged" | "liked" | "viewed";

/**
 * Record a behavioral impression.
 * Called server-side from operations (event page load, space visit, etc.).
 * Fire-and-forget — never throws, never blocks the caller.
 */
export function recordImpression(
  userId: string,
  itemId: string,
  itemType: ItemType,
  action: Action,
): void {
  db.insert(impressions)
    .values({ userId, itemId, itemType, action })
    .catch(() => {});
}
