/**
 * Assembler - Transforms raw assessment answers into structured ProfileData.
 * 
 * Processing Logic:
 * - Closed questions: Maps the 1-5 scalar value to the corresponding pre-defined embedding sentence.
 * - Open questions: Injects the user's response into the question's sentence template.
 * 
 * Output: Four textual descriptions (one for each matching axis).
 */

import type { AssessmentAnswersJson } from "./schema";
import { QUESTIONS, type Section, type OpenQuestion } from "./questions";
import type { ProfileData } from "@/lib/models/profiles/operations";

// ============================================
// ASSEMBLY LOGIC
// ============================================

function assembleSection(section: Section, answers: AssessmentAnswersJson): string {
  const questions = QUESTIONS[section];
  const sentences: string[] = [];

  for (const question of questions) {
    const answer = answers[question.id];
    if (answer === undefined) continue;

    if (question.type === "closed") {
      if (typeof answer === "number") {
        // Map scale 1-5 to array index 0-4
        const index = Math.max(0, Math.min(4, answer - 1));
        const sentence = question.options[index];
        if (sentence) sentences.push(sentence);
      } else if (typeof answer === "string") {
        // Use direct string if it matches a valid option
        if (question.options.includes(answer)) {
          sentences.push(answer);
        }
      }
    }

    if (question.type === "open" && typeof answer === "string") {
      const text = answer.trim();
      if (text) {
        // Inject answer into the predefined sentence template
        const openQ = question as OpenQuestion;
        const sentence = openQ.template.replace("{answer}", text);
        sentences.push(sentence);
      }
    }
  }

  return sentences.join(". ") + ".";
}

// ============================================
// EXPORT
// ============================================

/**
 * Aggregates all validated answers into a ProfileData object.
 * 
 * @param answers - Map of { questionId: value } (1-5 for closed, string for open).
 * 
 * @example
 * const answers = {
 *   "psy-1": 5,
 *   "psy-open": "curious and reflective",
 *   ...
 * };
 * 
 * const profile = assembleProfile(answers);
 */
export function assembleProfile(answers: AssessmentAnswersJson): ProfileData {
  return {
    psychologicalDesc: assembleSection("psychological", answers),
    valuesDesc: assembleSection("values", answers),
    interestsDesc: assembleSection("interests", answers),
    behavioralDesc: assembleSection("behavioral", answers),
  };
}
