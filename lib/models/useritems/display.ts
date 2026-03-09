/**
 * Pure display utility — no DB, no side effects.
 *
 * Takes photos and prompts already sorted by their own displayOrder,
 * and interleaves them into a single sequence where:
 * - First and last positions are always photos
 * - Prompts are distributed evenly in the interior slots
 */
export function interleaveProfileItems<T extends { type: "photo" | "prompt" }>(
  items: T[],
): T[] {
  const photos = items.filter((i) => i.type === "photo");
  const prompts = items.filter((i) => i.type === "prompt");

  if (prompts.length === 0) return photos;
  if (photos.length === 0) return prompts;

  const N = photos.length;
  const M = prompts.length;

  // insertAfterPhotoIndex[k] = index of the photo after which prompt k is inserted
  // Distributes prompts uniformly in the interior, keeping first and last as photos
  const insertAfterPhotoIndex = prompts.map((_, k) =>
    Math.floor(((k + 1) * (N - 1)) / (M + 1))
  );

  const result: T[] = [];
  for (let photoIdx = 0; photoIdx < photos.length; photoIdx++) {
    result.push(photos[photoIdx]);
    // Insert all prompts whose insertion point is after this photo
    while (prompts.length > 0) {
      const nextPromptIdx = result.filter((i) => i.type === "prompt").length;
      if (nextPromptIdx < M && insertAfterPhotoIndex[nextPromptIdx] === photoIdx) {
        result.push(prompts[nextPromptIdx]);
      } else {
        break;
      }
    }
  }

  return result;
}
