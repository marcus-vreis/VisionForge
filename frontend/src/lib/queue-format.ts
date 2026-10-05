/** Display helpers for the run queue (ADR-075).
 *
 * The backend labels a job with the task key and the strategy it was submitted
 * under; both are internal identifiers, and a queue panel is exactly where a
 * researcher should not have to read `replicated-comparison` or `custom:foo`.
 *
 * The words come from the language dictionaries (`queueFormat` in src/i18n), so
 * the functions take the dictionary: `taskLabel(t, job.task)` with
 * `const t = useT()`.
 */

import type { Dict } from "../i18n/pt";

/** The dictionary entry each strategy the backend sends is shown as. */
const STRATEGY_KEYS: Record<string, keyof Dict["queueFormat"]["strategies"]> = {
  simple: "simple",
  // Classification submits its strategy as config.block, so the plain path
  // arrives under its block name rather than "simple".
  classification: "simple",
  cross_validation: "kfold",
  cv: "kfold",
  transfer_learning: "transferLearning",
  grid_search: "gridSearch",
  random_search: "randomSearch",
  sweep: "sweep",
  replicates: "replicates",
  comparison: "comparison",
  "replicated-comparison": "replicatedComparison",
};

/** `custom:counting` renders as the researcher's own task name. */
export function taskLabel(t: Dict, task: string): string {
  if (task.startsWith("custom:")) return task.slice("custom:".length);
  const labels: Record<string, string> = t.queueFormat.tasks;
  return labels[task] ?? task;
}

export function strategyLabel(t: Dict, strategy: string): string {
  const labels = t.queueFormat.strategies;
  if (strategy.startsWith("sweep:")) {
    return `${labels.sweep} · ${strategy.slice("sweep:".length)}`;
  }
  const key = STRATEGY_KEYS[strategy];
  return key ? labels[key] : strategy;
}

/** How long a job has been waiting, in the coarsest unit that still reads. */
export function waitedFor(submittedAt: string, now: number = Date.now()): string {
  const seconds = Math.max(0, (now - Date.parse(submittedAt)) / 1000);
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}min`;
  return `${(seconds / 3600).toFixed(1)}h`;
}
