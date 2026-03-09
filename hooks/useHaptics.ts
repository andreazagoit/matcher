"use client";

import { useWebHaptics } from "web-haptics/react";
import type { HapticInput } from "web-haptics";

export const hapticPatterns = {
  tap:         "light"       as const, // navigazione, segna letta, selezione
  confirm:     "selection"   as const, // aggiunta elemento, toggle pill
  success:     "success"     as const, // salva, join space, RSVP
  send:        [{ duration: 15 }, { delay: 30, duration: 20, intensity: 0.7 }],
  delete:      [{ duration: 40, intensity: 0.9 }, { delay: 50, duration: 20, intensity: 0.5 }],
  destructive: "error"       as const, // leave / delete irreversibile
  drop:        "medium"      as const, // drag & drop rilascio
} satisfies Record<string, HapticInput>;

export function useHaptics() {
  const { trigger } = useWebHaptics();
  return (pattern: HapticInput) => trigger(pattern);
}
