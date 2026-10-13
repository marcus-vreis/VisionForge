/** What the browser says about whether the drawing is being looked at, and how
 * much motion the reader accepts (ADR-118). The decision itself is
 * `storyMotion` in lib/training-story.ts; these only read the two facts.
 */

import { useSyncExternalStore } from "react";

const REDUCED = "(prefers-reduced-motion: reduce)";

function subscribeReduced(onChange: () => void): () => void {
  const query = window.matchMedia(REDUCED);
  query.addEventListener("change", onChange);
  return () => query.removeEventListener("change", onChange);
}

/** The reader asked the system for less motion. Without a DOM it is `false`. */
export function usePrefersReducedMotion(): boolean {
  return useSyncExternalStore(
    subscribeReduced,
    () => window.matchMedia(REDUCED).matches,
    () => false,
  );
}

function subscribeVisibility(onChange: () => void): () => void {
  document.addEventListener("visibilitychange", onChange);
  return () => document.removeEventListener("visibilitychange", onChange);
}

/** The browser tab is in the foreground. Without a DOM it is `true`. */
export function useDocumentVisible(): boolean {
  return useSyncExternalStore(
    subscribeVisibility,
    () => document.visibilityState !== "hidden",
    () => true,
  );
}
