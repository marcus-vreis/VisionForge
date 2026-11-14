/** What the run detail shows, in which order, and which number leads.
 *
 * The detail reads as a results page: the numbers and the charts first, then
 * what can be done with the run, and only then the reference material (where it
 * is on disk, the environment, the config), folded so it does not push the
 * results off the first screen.
 */

/** One block of the run detail. */
export type DetailSection =
  | "member"
  | "group"
  | "metrics"
  | "charts"
  | "actions"
  | "tests"
  | "dataset"
  | "location"
  | "environment"
  | "training"
  | "pipeline";

/** The reference blocks, folded, after everything else. */
export const REFERENCE_SECTIONS = [
  "dataset",
  "location",
  "environment",
  "training",
  "pipeline",
] as const satisfies readonly DetailSection[];

export type ReferenceSection = (typeof REFERENCE_SECTIONS)[number];

/** The blocks of a run's detail, top to bottom.
 *
 * A group (ADR-113) is a job, not a training: its own summary leads, and it has
 * no checkpoint to test. A seed of a group says so before its numbers.
 */
export function detailSections(detail: {
  group?: unknown;
  group_id?: string | null;
}): DetailSection[] {
  if (detail.group) return ["group", "metrics", "charts", "actions", ...REFERENCE_SECTIONS];
  return [
    ...(detail.group_id ? (["member"] as const) : []),
    "metrics",
    "charts",
    "actions",
    "tests",
    ...REFERENCE_SECTIONS,
  ];
}

/** Whether a reference block starts unfolded: the short, often-read ones do;
 *  paths, library versions and the pipeline's knobs wait for a click. */
export function sectionStartsOpen(section: ReferenceSection): boolean {
  return section === "dataset" || section === "training";
}

// The number each task's History card leads with (`_SUMMARY_METRIC_KEYS` in
// run_history.py), in the run.json names it may be stored under: the held-out
// split first, then the validation score a run without one falls back to.
const HEADLINE_KEYS: Record<string, readonly string[]> = {
  classification: ["test_accuracy", "val_accuracy", "accuracy", "best_val_accuracy"],
  detection: ["map50", "map50_95"],
  regression: ["test_r2", "val_r2", "r2"],
  segmentation: ["test_miou", "val_miou", "miou"],
  anomaly: ["test_auroc", "val_auroc", "auroc"],
};

// Bookkeeping, never a result.
const NOT_A_RESULT = new Set(["total_epochs", "best_epoch"]);

function finite(value: unknown): boolean {
  return typeof value === "number" && Number.isFinite(value);
}

/** The metric the detail emphasises, or null when the run measured none of its
 *  task's headline numbers. A researcher's task (`custom:<key>`) leads with the
 *  first number it reported; an unknown task is read as classification, as the
 *  server reads it. */
export function headlineMetricKey(
  task: string,
  metrics: Readonly<Record<string, unknown>>,
): string | null {
  if (task.startsWith("custom:")) {
    return (
      Object.keys(metrics).find((key) => !NOT_A_RESULT.has(key) && finite(metrics[key])) ?? null
    );
  }
  const keys = HEADLINE_KEYS[task] ?? HEADLINE_KEYS["classification"];
  return keys.find((key) => finite(metrics[key])) ?? null;
}
