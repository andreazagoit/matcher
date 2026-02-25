/**
 * Prompt metadata for profile items.
 * Question strings live in messages/it.json under "prompts.*".
 * This file only contains structural metadata (key, category).
 */

export type PromptCategory = "personalita" | "stile_di_vita" | "relazioni" | "creativita" | "umorismo";

export interface PromptDefinition {
  key: string;
  category: PromptCategory;
}

export const PROMPTS: PromptDefinition[] = [
  // ── Personalità ──────────────────────────────────────────
  { key: "controversial_opinion", category: "personalita" },
  { key: "best_quality", category: "personalita" },
  { key: "i_am_always", category: "personalita" },
  { key: "surprising_fact", category: "personalita" },
  { key: "love_language", category: "personalita" },
  { key: "core_value", category: "personalita" },

  // ── Stile di vita ─────────────────────────────────────────
  { key: "sunday_plan", category: "stile_di_vita" },
  { key: "cant_live_without", category: "stile_di_vita" },
  { key: "recent_discovery", category: "stile_di_vita" },
  { key: "life_goal", category: "stile_di_vita" },
  { key: "obsessed_with", category: "stile_di_vita" },
  { key: "my_routine", category: "stile_di_vita" },

  // ── Relazioni ─────────────────────────────────────────────
  { key: "win_me_over", category: "relazioni" },
  { key: "looking_for", category: "relazioni" },
  { key: "relationship_green_flag", category: "relazioni" },
  { key: "perfect_first_date", category: "relazioni" },
  { key: "communication_style", category: "relazioni" },
  { key: "dealbreaker", category: "relazioni" },

  // ── Creatività ────────────────────────────────────────────
  { key: "dinner_with_anyone", category: "creativita" },
  { key: "hidden_talent", category: "creativita" },
  { key: "two_truths_lie", category: "creativita" },
  { key: "bucket_list", category: "creativita" },
  { key: "alter_ego", category: "creativita" },
  { key: "unpopular_take", category: "creativita" },

  // ── Umorismo ──────────────────────────────────────────────
  { key: "most_spontaneous", category: "umorismo" },
  { key: "innocent_red_flag", category: "umorismo" },
  { key: "guilty_pleasure", category: "umorismo" },
  { key: "worst_habit", category: "umorismo" },
  { key: "emoji_story", category: "umorismo" },
  { key: "what_friends_say", category: "umorismo" },
];

export const PROMPTS_BY_KEY: Record<string, PromptDefinition> = Object.fromEntries(
  PROMPTS.map((p) => [p.key, p])
);

export const PROMPTS_BY_CATEGORY: Record<PromptCategory, PromptDefinition[]> = {
  personalita: [],
  stile_di_vita: [],
  relazioni: [],
  creativita: [],
  umorismo: [],
};
for (const p of PROMPTS) {
  PROMPTS_BY_CATEGORY[p.category].push(p);
}

export const CATEGORY_LABELS: Record<PromptCategory, string> = {
  personalita: "Personalità",
  stile_di_vita: "Stile di vita",
  relazioni: "Relazioni",
  creativita: "Creatività",
  umorismo: "Umorismo",
};
