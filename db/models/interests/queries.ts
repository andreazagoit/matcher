import { INTERESTS_OPTIONS, type Interest } from "./data";

/**
 * Ottieni tutti gli interessi disponibili
 */
export function getAllInterests(): Interest[] {
  return [...INTERESTS_OPTIONS];
}

/**
 * Verifica se un interesse è valido
 */
export function isValidInterest(interest: string): interest is Interest {
  return INTERESTS_OPTIONS.includes(interest as Interest);
}

/**
 * Ottieni interesse per ID (se necessario in futuro)
 */
export function getInterestById(id: string): Interest | null {
  const index = parseInt(id, 10);
  if (isNaN(index) || index < 0 || index >= INTERESTS_OPTIONS.length) {
    return null;
  }
  return INTERESTS_OPTIONS[index];
}

