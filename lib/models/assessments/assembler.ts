/**
 * Assembler - Converte risposte → ProfileData
 * 
 * Input: { questionId: valore }
 * - Chiuse: indice 1-5 → prende l'opzione corrispondente (self-contained)
 * - Aperte: stringa → applica il template della domanda
 * 
 * Output: 4 descrizioni testuali (una per asse)
 */

import type { AssessmentAnswersJson } from "./schema";
import { QUESTIONS, type Section, type OpenQuestion } from "./questions";
import type { ProfileData } from "@/lib/models/profiles/operations";

// ============================================
// ASSEMBLAGGIO
// ============================================

function assembleSection(section: Section, answers: AssessmentAnswersJson): string {
  const questions = QUESTIONS[section];
  const sentences: string[] = [];

  for (const question of questions) {
    const answer = answers[question.id];
    if (answer === undefined) continue;

    if (question.type === "closed") {
      if (typeof answer === "number") {
        // Valore 1-5 → indice 0-4
        const index = Math.max(0, Math.min(4, answer - 1));
        const sentence = question.options[index];
        if (sentence) sentences.push(sentence);
      } else if (typeof answer === "string") {
        // Stringa diretta dell'opzione selezionata
        if (question.options.includes(answer)) {
          sentences.push(answer);
        }
      }
    }

    if (question.type === "open" && typeof answer === "string") {
      const text = answer.trim();
      if (text) {
        // Applica il template della domanda
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
 * Assembla le risposte in ProfileData
 * 
 * @param answers - { questionId: valore } (1-5 per chiuse, stringa per aperte)
 * 
 * @example
 * const answers = {
 *   "psy-1": 5,        // → prende options[4]
 *   "psy-open": "curiosa e riflessiva",  // → "Mi descrivo come una persona curiosa e riflessiva"
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
