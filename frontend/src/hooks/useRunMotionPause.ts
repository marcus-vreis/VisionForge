import { useEffect } from "react";

/** Marks `root` for the stylesheet rule and returns how to clear the mark. */
export function markRunActive(root: {
  dataset: Record<string, string | undefined>;
}): () => void {
  root.dataset.runActive = "true";
  return () => {
    delete root.dataset.runActive;
  };
}

/**
 * Marks the page while a job runs so `index.css` ends every infinite animation:
 * any one of them keeps the browser producing frames on the GPU that is training,
 * which costs about 17% of epoch time (ADR-157).
 */
export function useRunMotionPause(active: boolean): void {
  useEffect(() => {
    if (!active) return;
    return markRunActive(document.documentElement);
  }, [active]);
}
