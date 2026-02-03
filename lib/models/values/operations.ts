import { VALUES_OPTIONS, type Value } from "./data";

export { VALUES_OPTIONS, type Value };

/**
 * Ottieni tutti i valori disponibili
 */
export function getAllValues(): Value[] {
  return [...VALUES_OPTIONS];
}

/**
 * Verifica se un valore è valido
 */
export function isValidValue(value: string): value is Value {
  return VALUES_OPTIONS.includes(value as Value);
}

/**
 * Ottieni valori per ID (se necessario in futuro)
 */
export function getValueById(id: string): Value | null {
  const index = parseInt(id, 10);
  if (isNaN(index) || index < 0 || index >= VALUES_OPTIONS.length) {
    return null;
  }
  return VALUES_OPTIONS[index];
}

