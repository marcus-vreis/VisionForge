import type { Theme } from "./theme";

/** The color each task goes by in the interface: classification red, detection
 *  green, regression blue, segmentation violet (anomaly amber). The history list
 *  and a run's detail read it from here, so a run is the same color wherever it
 *  appears. */
export const TASK_ACCENT: Record<string, string> = {
  classification: "oklch(0.74 0.18 22)",
  detection: "oklch(0.78 0.18 150)",
  regression: "oklch(0.74 0.16 240)",
  segmentation: "oklch(0.74 0.18 305)",
  anomaly: "oklch(0.80 0.15 75)",
};

/** The same colors for the light theme, as index.css darkens the tabs' accents
 *  (ADR-125); anomaly, which has no tab color there, is mirrored around 0.65. */
export const TASK_ACCENT_LIGHT: Record<string, string> = {
  classification: "oklch(0.56 0.18 22)",
  detection: "oklch(0.52 0.15 150)",
  regression: "oklch(0.52 0.16 240)",
  segmentation: "oklch(0.52 0.18 305)",
  anomaly: "oklch(0.50 0.15 75)",
};

const FALLBACK = "var(--vf-text-muted)";

/** The task colors of a theme. They stay literals in both: the training story
 *  sets them as SVG attributes and the history appends alpha to them. */
export function taskAccents(theme: Theme = "dark"): Record<string, string> {
  return theme === "light" ? TASK_ACCENT_LIGHT : TASK_ACCENT;
}

/** The accent of a task as a run records it: classification keeps its problem
 *  type (`binary`, `multiclass`) where the others keep their own name, and a
 *  researcher's own task (`custom:<key>`) has no color of its own. */
export function accentForTask(task: string | null | undefined, theme: Theme = "dark"): string {
  if (!task || task.startsWith("custom:")) return FALLBACK;
  const accents = taskAccents(theme);
  if (task in accents && task !== "classification") return accents[task];
  return accents.classification;
}
