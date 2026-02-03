/**
 * Assembler - Converte risposte → ProfileData
 * 
 * Input: { questionId: valore }
 * - Chiuse: valore 1-5
 * - Aperte: stringa
 */

import type { ProfileAxis } from "@/lib/models/profiles/schema";
import type { TestAnswersJson } from "./schema";
import { QUESTIONS, SECTIONS, type Section, type Question } from "./questions";

// ============================================
// OUTPUT
// ============================================

export interface ProfileData {
  psychological: ProfileAxis;
  values: ProfileAxis;
  interests: ProfileAxis;
  behavioral: ProfileAxis;
}

// ============================================
// ASSEMBLAGGIO
// ============================================

function assembleSection(section: Section, answers: TestAnswersJson): ProfileAxis {
  const questions = QUESTIONS[section];
  const sentences: string[] = [];

  for (const question of questions) {
    const answer = answers[question.id];
    if (answer === undefined) continue;

    if (question.type === "closed" && typeof answer === "number") {
      // Valore 1-5 → indice 0-4
      const index = Math.max(0, Math.min(4, answer - 1));
      const sentence = question.options[index];
      if (sentence) sentences.push(sentence);
    }

    if (question.type === "open" && typeof answer === "string") {
      const text = answer.trim();
      if (text) sentences.push(text);
    }
  }

  const description = sentences.join(". ") + ".";

  return {
    traits: {},
    description,
  };
}

// ============================================
// EXPORT
// ============================================

/**
 * Assembla le risposte in ProfileData
 * 
 * @param answers - { questionId: valore } (1-5 per chiuse, stringa per aperte)
 * 
 * @example
 * const answers = {
 *   "psy-1": 5,        // "Amo conoscere gente nuova..."
 *   "psy-2": 1,        // "Ho bisogno di stare solo/a..."
 *   "psy-open": "Sono curioso e riflessivo",
 *   "val-1": 4,
 *   ...
 * };
 * 
 * const profile = assembleProfile(answers);
 */
export function assembleProfile(answers: TestAnswersJson): ProfileData {
  return {
    psychological: assembleSection("psychological", answers),
    values: assembleSection("values", answers),
    interests: assembleSection("interests", answers),
    behavioral: assembleSection("behavioral", answers),
  };
}
