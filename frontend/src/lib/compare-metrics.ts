/**
 * Which metrics the History comparison puts in its table, per task.
 *
 * Every task writes its own names into run.json `metrics` (a detection run has
 * `map50`, a regression run `test_r2`, a researcher's own task whatever it
 * reports), so one fixed list of classification names shows dashes for all the
 * others. The rows come from the task of the runs being compared; runs of
 * different tasks are never put in one table (`distinctTasks`).
 *
 * Pure, and free of text: a row carries the dictionary key of its label
 * (`compareRuns.metrics`), or none when the metric is a researcher's own and
 * its name is the label.
 *
 * The History card names its few headline numbers from the same table
 * (`cardMetrics`), so a task's metrics are listed in one place.
 */
import type { Dict } from "../i18n/pt";

export type MetricDirection = "higher" | "lower";

/** Keys of `compareRuns.metrics`: the dictionary's name for a built-in row. */
export type MetricLabelKey = keyof Dict["compareRuns"]["metrics"];

export interface CompareMetric {
  /** The name in run.json `metrics`. */
  key: string;
  /** The dictionary's label; null where the metric's own name is the label. */
  label: MetricLabelKey | null;
  /** Which end is better. Null for a count or a decision point, which is
   *  neither: nothing to highlight in that row. */
  direction: MetricDirection | null;
}

const STANDALONE_TASKS = ["detection", "regression", "segmentation", "anomaly"];

/** Substrings that mark a metric as lower-is-better: the backend's rule
 *  (`infer_direction` in core/significance.py), copied rather than reinvented so
 *  a metric reads the same way in a ranking, a sweep and this table. */
const LOWER_IS_BETTER = ["loss", "mae", "mse", "rmse", "error", "err", "distance"];

/** Bookkeeping that every run reports and that is no better for being larger. */
const BOOKKEEPING: MetricLabelKey[] = ["best_epoch", "total_epochs"];

/** Metrics that are a decision point or a count, not a quality. */
const NEUTRAL = new Set<string>([...BOOKKEEPING, "threshold", "test_threshold"]);

export function inferMetricDirection(name: string): MetricDirection {
  const lowered = name.toLowerCase();
  return LOWER_IS_BETTER.some((token) => lowered.includes(token)) ? "lower" : "higher";
}

/**
 * The rows of each built-in task, in display order: the training bookkeeping the
 * table always had, then what the task is judged by. A `test_` row is the held-out
 * split; the bare name is the validation score at the best epoch, which is all a
 * run that never evaluated a test split has (and for detection all there is).
 */
const BUILTIN_ROWS: Record<string, MetricLabelKey[]> = {
  classification: [
    "best_val_loss",
    "best_epoch",
    "total_epochs",
    "test_accuracy",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_auc_roc",
  ],
  detection: ["best_epoch", "total_epochs", "map50_95", "map50", "precision", "recall"],
  regression: [
    "best_val_loss",
    "best_epoch",
    "total_epochs",
    "test_r2",
    "test_rmse",
    "test_mae",
    "r2",
    "rmse",
    "mae",
  ],
  segmentation: [
    "best_epoch",
    "total_epochs",
    "test_miou",
    "test_dice",
    "test_pixel_acc",
    "miou",
    "dice",
    "pixel_acc",
  ],
  anomaly: [
    "best_epoch",
    "total_epochs",
    "test_auroc",
    "test_image_f1",
    "test_threshold",
    "auroc",
    "image_f1",
    "threshold",
  ],
};

/** A number a table can print and compare; null for anything else (a metric
 *  never measured is null in run.json, a degenerate one NaN or infinite). */
export function numericMetric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** The task family of a run: the server's answer, else the rule it applies to
 *  run.json (`config.task` for the standalone tasks, classification otherwise). */
export function runTaskKey(run: {
  task?: string;
  config: Record<string, unknown>;
}): string {
  if (typeof run.task === "string" && run.task !== "") return run.task;
  const task = run.config.task;
  return typeof task === "string" && STANDALONE_TASKS.includes(task) ? task : "classification";
}

/** The distinct tasks among the runs, in the order they first appear. */
export function distinctTasks(
  runs: ReadonlyArray<{ task?: string; config: Record<string, unknown> }>,
): string[] {
  return [...new Set(runs.map(runTaskKey))];
}

export function isCustomTaskKey(task: string): boolean {
  return task.startsWith("custom:");
}

function directionOf(
  key: string,
  declared: Readonly<Record<string, string>> | undefined,
): MetricDirection | null {
  if (NEUTRAL.has(key)) return null;
  // A task that declared its metrics is believed over the name, as in the backend.
  const stated = declared?.[key];
  if (stated === "higher" || stated === "lower") return stated;
  return inferMetricDirection(key);
}

/**
 * The table rows for runs of one `task`: only the metrics at least one run
 * actually measured, so a task never shows the rows of another.
 *
 * `declared` is a researcher's task's own statement of which way each metric
 * improves (`GET /api/tasks` -> `metrics`); a built-in task ignores it. A task
 * that is not built in lists the bookkeeping and then every numeric metric its
 * runs reported, under the names they reported them.
 */
export function metricRows(
  task: string,
  runs: ReadonlyArray<{ metrics: Record<string, unknown> }>,
  declared?: Readonly<Record<string, string>>,
): CompareMetric[] {
  const measured = (key: string) => runs.some((run) => numericMetric(run.metrics[key]) !== null);

  const builtin = BUILTIN_ROWS[task];
  if (builtin) {
    return builtin
      .filter(measured)
      .map((key) => ({ key, label: key, direction: directionOf(key, undefined) }));
  }

  const reported = runs.flatMap((run) =>
    Object.keys(run.metrics).filter((key) => numericMetric(run.metrics[key]) !== null),
  );
  const keys = [...new Set([...BOOKKEEPING, ...reported])];
  return keys
    .filter(measured)
    .map((key) => ({
      key,
      // The bookkeeping is the engine's own, so the dictionary names it; the rest is
      // the researcher's vocabulary and keeps their spelling.
      label: BOOKKEEPING.find((name) => name === key) ?? null,
      direction: directionOf(key, declared),
    }));
}

/** How many headline numbers a History card prints. */
const CARD_METRIC_LIMIT = 3;

export interface CardMetric {
  /** The name in the run's summary (`RunSummary.final_metrics`). */
  key: string;
  /** The table row it is read from, whose label the card prints; null where the
   *  metric's own name is the label (a researcher's task, or a name the table
   *  does not know). */
  label: MetricLabelKey | null;
}

/** What the server puts before a headline's name when it is the validation score
 *  (`routes._summary_metrics`). */
const VALIDATION_PREFIX = "val_";

/**
 * The row of a task's table that a summary name stands for. The server projects
 * a run onto a few short names (`r2`, `miou`, `f1`), each one read from the
 * held-out split's `test_<name>` where the task has one (and from the bare name
 * where it does not, like detection's `map50`). A row answers to a name when it
 * is that name or ends in `_<name>` (`test_image_f1` is anomaly's `f1`), and
 * when two do, the held-out one wins: it is the number the server read.
 *
 * A run that never scored a test split has no `test_<name>`; the server then
 * sends the validation score as `val_<name>`. That name is answered by the bare
 * (validation) row, never a `test_` one: the card must not call it "(teste)".
 */
function rowOfSummaryName(rows: readonly MetricLabelKey[], name: string): MetricLabelKey | null {
  const answering = (wanted: string) =>
    rows.filter((row) => !NEUTRAL.has(row) && (row === wanted || row.endsWith(`_${wanted}`)));
  const own = answering(name);
  const held = own.find((row) => row.startsWith("test_")) ?? own[0];
  if (held !== undefined) return held;
  if (!name.startsWith(VALIDATION_PREFIX)) return null;
  const bare = answering(name.slice(VALIDATION_PREFIX.length));
  return bare.find((row) => !row.startsWith("test_")) ?? null;
}

/**
 * The headline numbers of a run's History card, in the order the server sent
 * them (its projection of run.json is the headline set), at most three.
 *
 * Nothing is listed here per task: the names come from the run, and the label of
 * each from the same table the comparison uses, so a task added to
 * `BUILTIN_ROWS` is labelled on the card and in the comparison at once. A
 * researcher's own task keeps the names it reported. A task this table does not
 * know is read as classification, which is what the server projects for it.
 * A metric never measured (null, NaN, infinite) is left out; zero is a reading.
 */
export function cardMetrics(
  task: string,
  finalMetrics: Readonly<Record<string, unknown>>,
): CardMetric[] {
  const rows = isCustomTaskKey(task) ? null : (BUILTIN_ROWS[task] ?? BUILTIN_ROWS.classification);
  return Object.keys(finalMetrics)
    .filter((key) => numericMetric(finalMetrics[key]) !== null)
    .slice(0, CARD_METRIC_LIMIT)
    .map((key) => ({ key, label: rows ? rowOfSummaryName(rows, key) : null }));
}

/**
 * Where a row's extreme value is: the indices holding the highest value of a
 * higher-is-better metric or the lowest of a lower-is-better one.
 *
 * Empty when there is nothing to tell apart: no direction, fewer than two
 * measured values, or all of them equal. Ties all count. The extreme is a
 * description of the numbers, not a verdict: with one seed per run it can be
 * the seed.
 */
export function extremeIndexes(
  values: ReadonlyArray<number | null>,
  direction: MetricDirection | null,
): number[] {
  if (direction === null) return [];
  const measured = values.filter((v): v is number => v !== null);
  if (measured.length < 2) return [];
  const high = Math.max(...measured);
  const low = Math.min(...measured);
  if (high === low) return [];
  const target = direction === "higher" ? high : low;
  return values.flatMap((v, i) => (v === target ? [i] : []));
}
