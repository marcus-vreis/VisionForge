/** Display helpers for the run queue (ADR-075).
 *
 * The backend labels a job with the task key and the strategy it was submitted
 * under; both are internal identifiers, and a queue panel is exactly where a
 * researcher should not have to read `replicated-comparison` or `custom:foo`.
 */

const TASK_LABELS: Record<string, string> = {
  classification: "Classificação",
  detection: "Detecção",
  regression: "Regressão",
  segmentation: "Segmentação",
  anomaly: "Anomalia",
};

const STRATEGY_LABELS: Record<string, string> = {
  simple: "treino simples",
  // Classification submits its strategy as config.block, so the plain path
  // arrives under its block name rather than "simple".
  classification: "treino simples",
  cross_validation: "K-fold",
  cv: "K-fold",
  transfer_learning: "transfer learning",
  grid_search: "grid search",
  random_search: "random search",
  sweep: "sweep",
  replicates: "réplicas",
  comparison: "comparação",
  "replicated-comparison": "comparação replicada",
};

/** `custom:counting` renders as the researcher's own task name. */
export function taskLabel(task: string): string {
  if (task.startsWith("custom:")) return task.slice("custom:".length);
  return TASK_LABELS[task] ?? task;
}

export function strategyLabel(strategy: string): string {
  if (strategy.startsWith("sweep:")) {
    return `sweep · ${strategy.slice("sweep:".length)}`;
  }
  return STRATEGY_LABELS[strategy] ?? strategy;
}

/** How long a job has been waiting, in the coarsest unit that still reads. */
export function waitedFor(submittedAt: string, now: number = Date.now()): string {
  const seconds = Math.max(0, (now - Date.parse(submittedAt)) / 1000);
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}min`;
  return `${(seconds / 3600).toFixed(1)}h`;
}
