/**
 * The History ranking: runs of one task on one dataset, ordered by one metric.
 *
 * A number is only comparable to another when both runs saw the same data and
 * were scored the same way, so nothing is ranked across tasks, across datasets or
 * across metrics. This file decides what counts as "the same dataset", which
 * metric a board opens on, the order, and when that order must not be trusted.
 *
 * Nothing here is a statistic. A replicate group (ADR-113) arrives with its mean
 * and its 95% interval already computed by the report; the only arithmetic is
 * sorting, and the one check on the intervals is whether two given intervals
 * overlap. The order is a description of the numbers, never a verdict (ADR-112):
 * the first row is "1st by mean" or "1st by value", not "the best", and the board
 * says so itself when the gap rests on one seed apiece or on intervals that
 * overlap. Pure and free of text: the words come from the dictionary.
 */
import type { MetricAggregate, RunSummary } from "../types/run";
import {
  cardMetrics,
  metricDirection,
  numericMetric,
  servedDirections,
  tableRowsOf,
  type CardMetric,
  type MetricDirection,
} from "./compare-metrics";
import type { HistoryEntry } from "./run-groups";

// ─── which runs saw the same data ────────────────────────────────────────────

/** How a run's dataset is told apart from another's: by a fingerprint that
 *  covers the files, by the path alone, or not at all. */
export type DatasetBasis = "fingerprint" | "path" | "unknown";

export interface DatasetIdentity {
  /** Equal keys mean the same dataset; the key of every unknown run is the same
   *  string, which is what puts them in one bucket. */
  key: string;
  basis: DatasetBasis;
}

/**
 * A dataset path in one spelling, so `C:\data\Coffee\` and `c:/data/coffee` are
 * one dataset. Separators and trailing slashes always; case only on a Windows
 * drive path, whose file system ignores it (on any other path two names that
 * differ by case are two folders).
 */
export function normalizeDatasetPath(path: string): string {
  let out = path.trim().replace(/\\/g, "/");
  // Collapse repeated separators but keep a leading `//` (a network share).
  out = out.replace(/(?!^)\/{2,}/g, "/");
  if (out.length > 1) out = out.replace(/\/+$/, "");
  return /^[a-z]:/i.test(out) ? out.toLowerCase() : out;
}

/**
 * The dataset a run trained on, as far as the list can prove it. The fingerprint
 * wins: it covers the files, so two runs under one path that saw different data
 * stay apart. Without one (a run older than ADR-061, or whose fingerprint was
 * unavailable) the path stands in, and the board says it is only the path. A run
 * with neither belongs to no dataset we can name.
 *
 * A path-only run is never merged into a fingerprinted board, even when the
 * paths match: the path does not say the files did not change in between.
 */
export function datasetIdentity(
  run: Pick<RunSummary, "dataset_digest" | "dataset_method" | "dataset_root">,
): DatasetIdentity {
  if (run.dataset_digest) {
    return { key: `fp:${run.dataset_method ?? ""}:${run.dataset_digest}`, basis: "fingerprint" };
  }
  const root = run.dataset_root?.trim();
  if (root) return { key: `path:${normalizeDatasetPath(root)}`, basis: "path" };
  return { key: "unknown", basis: "unknown" };
}

// ─── boards: one per (task, dataset) ─────────────────────────────────────────

export interface Board {
  /** `<task>|<dataset key>`: stable across reloads, so a picked metric sticks. */
  id: string;
  /** The raw `run.task`, not the family: a binary and a multiclass problem on
   *  the same folder do not measure the same thing. */
  task: string;
  identity: DatasetIdentity;
  /** The dataset as the newest run of the board names it. */
  name: string | null;
  root: string | null;
  digest: string | null;
  method: string | null;
  /** History's entries (a replicate group is one, its seeds folded under it). */
  entries: HistoryEntry[];
}

/**
 * The entries split into boards by (task, dataset identity), in the order each
 * board first appears (the list is newest first), the boards of an unknown
 * dataset last. An unknown board lists its runs and ranks none.
 */
export function groupBoards(entries: ReadonlyArray<HistoryEntry>): Board[] {
  const boards = new Map<string, Board>();
  for (const entry of entries) {
    const { run } = entry;
    const identity = datasetIdentity(run);
    const id = `${run.task}|${identity.key}`;
    const board = boards.get(id);
    if (board) {
      board.entries.push(entry);
      board.name ??= run.dataset_name ?? null;
      continue;
    }
    boards.set(id, {
      id,
      task: run.task,
      identity,
      name: run.dataset_name ?? null,
      root: run.dataset_root ?? null,
      digest: run.dataset_digest ?? null,
      method: run.dataset_method ?? null,
      entries: [entry],
    });
  }
  const all = [...boards.values()];
  return [
    ...all.filter((b) => b.identity.basis !== "unknown"),
    ...all.filter((b) => b.identity.basis === "unknown"),
  ];
}

// ─── which metric ────────────────────────────────────────────────────────────

export interface BoardMetric extends CardMetric {
  direction: MetricDirection;
}

/** The row that is the training objective on the validation split. A loss is a
 *  fine thing to read and a poor thing to rank on by default. */
const LOSS_ROW = "best_val_loss";

/**
 * The metrics a board can be ranked on: those its runs reported (the History
 * list's headline numbers), in the order of the task's metric table so the
 * held-out quality rows come first, then whatever the table does not place, in
 * the order the runs reported it. The direction is the server's
 * (`metric_directions`), else the metric's name.
 *
 * `accuracy` (held-out split) and `val_accuracy` (the validation score a run
 * with no test split falls back to) are different metrics and stay apart: a
 * board never ranks a validation number against a test one.
 */
export function boardMetrics(entries: ReadonlyArray<HistoryEntry>): BoardMetric[] {
  const runs = entries.map((entry) => entry.run);
  if (runs.length === 0) return [];
  const rows = tableRowsOf(runs[0].task);
  const served = servedDirections(runs);

  const seen = new Map<string, CardMetric>();
  for (const run of runs) {
    for (const metric of cardMetrics(run.task, run.final_metrics)) {
      if (!seen.has(metric.key)) seen.set(metric.key, metric);
    }
  }
  const position = (metric: CardMetric) =>
    rows && metric.label ? rows.indexOf(metric.label) : Number.POSITIVE_INFINITY;
  return [...seen.values()]
    .map((metric, order) => ({ metric, order }))
    .sort((a, b) => position(a.metric) - position(b.metric) || a.order - b.order)
    .map(({ metric }) => ({ ...metric, direction: metricDirection(metric.key, served) }));
}

/**
 * The metric a board opens on: the first quality row of its task's table that
 * its runs reported, which is the held-out score where the task has one (a
 * detection run has only the validation mAP). The validation loss is skipped
 * unless it is all there is.
 */
export function defaultMetric(metrics: ReadonlyArray<BoardMetric>): string | null {
  const preferred = metrics.find((metric) => metric.label !== LOSS_ROW) ?? metrics[0];
  return preferred?.key ?? null;
}

// ─── the order ───────────────────────────────────────────────────────────────

/** Why a run is listed but not ranked. */
export type UnrankedReason = "not-finished" | "stopped" | "comparison" | "no-metric";

export interface RankedRow {
  entry: HistoryEntry;
  /** Competition ranking: equal values share a rank (1, 1, 3). */
  rank: number;
  value: number;
  /** Trainings behind the value: 1 for a single run, the seeds that finished for
   *  a replicate group. */
  seeds: number;
  /** `mean` over two seeds or more, `value` of one run. */
  basis: "mean" | "value";
  /** The report's aggregate for the metric, when it has one (a group's mean, std
   *  and interval; read, not recomputed). */
  aggregate: MetricAggregate | null;
  /** The 95% interval the report gave; null on one seed or when it gave none. */
  low: number | null;
  high: number | null;
}

export interface UnrankedRow {
  entry: HistoryEntry;
  reason: UnrankedReason;
}

/** What the order cannot support, about its first two rows:
 *  - `single-seed`: both are one seed, so the gap can be the seed;
 *  - `single-seed-mixed`: one of them is, and has no interval;
 *  - `no-interval`: both are means over seeds but one carries no interval;
 *  - `overlap`: both carry an interval and the two intervals overlap. */
export type Caution = "single-seed" | "single-seed-mixed" | "no-interval" | "overlap";

export interface Ranking {
  ranked: RankedRow[];
  unranked: UnrankedRow[];
  caution: Caution | null;
}

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function unrankedReason(run: RunSummary, metric: string): UnrankedReason | null {
  if (run.status !== "completed") return "not-finished";
  if (run.stopped === true || run.group?.stopped === true) return "stopped";
  // A replicated comparison is several variants, not one number.
  if (run.group?.kind === "replicated_comparison") return "comparison";
  if (numericMetric(run.final_metrics[metric]) === null) return "no-metric";
  return null;
}

/** Whether two closed intervals share a point. */
export function intervalsOverlap(
  a: { low: number; high: number },
  b: { low: number; high: number },
): boolean {
  return a.low <= b.high && b.low <= a.high;
}

/**
 * Whether the order of the first two rows can be taken at face value. Null when
 * there is nothing to doubt, which is the only case in which the board says
 * nothing; it never claims the gap is real, only that it has not been shown to
 * be the seed's.
 */
export function rankingCaution(ranked: ReadonlyArray<RankedRow>): Caution | null {
  if (ranked.length < 2) return null;
  const [first, second] = ranked;
  const single = [first, second].filter((row) => row.seeds < 2).length;
  if (single === 2) return "single-seed";
  if (single === 1) return "single-seed-mixed";
  if (first.low === null || first.high === null || second.low === null || second.high === null) {
    return "no-interval";
  }
  return intervalsOverlap(
    { low: first.low, high: first.high },
    { low: second.low, high: second.high },
  )
    ? "overlap"
    : null;
}

/**
 * The board's entries ordered by `metric` in its `direction`, then the entries
 * that cannot be ranked with the reason each is left out. A replicate group is
 * ranked by its mean (the value the server keeps under the metric's name) and a
 * single run by its own number; a stopped run, one that did not finish, a
 * replicated comparison and a run that never measured the metric are listed
 * after, in the order they came.
 */
export function rankEntries(
  entries: ReadonlyArray<HistoryEntry>,
  metric: string,
  direction: MetricDirection,
): Ranking {
  const scored: Array<Omit<RankedRow, "rank">> = [];
  const unranked: UnrankedRow[] = [];

  for (const entry of entries) {
    const { run } = entry;
    const reason = unrankedReason(run, metric);
    const value = numericMetric(run.final_metrics[metric]);
    if (reason !== null || value === null) {
      unranked.push({ entry, reason: reason ?? "no-metric" });
      continue;
    }
    const group = run.group?.kind === "replicates" ? run.group : null;
    const aggregate = group?.final_aggregates?.[metric] ?? null;
    const seeds = group ? (aggregate?.n ?? group.n_finished) : 1;
    const low = seeds >= 2 ? finite(aggregate?.ci95_low) : null;
    const high = seeds >= 2 ? finite(aggregate?.ci95_high) : null;
    const interval = low !== null && high !== null;
    scored.push({
      entry,
      value,
      seeds,
      basis: seeds >= 2 ? "mean" : "value",
      aggregate,
      low: interval ? low : null,
      high: interval ? high : null,
    });
  }

  const sign = direction === "higher" ? -1 : 1;
  const ordered = [...scored].sort((a, b) => sign * (a.value - b.value));
  const ranked: RankedRow[] = [];
  ordered.forEach((row, index) => {
    const previous = ranked[index - 1];
    const rank = previous && previous.value === row.value ? previous.rank : index + 1;
    ranked.push({ ...row, rank });
  });
  return { ranked, unranked, caution: rankingCaution(ranked) };
}
