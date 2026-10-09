/**
 * Replicate groups in History (ADR-113).
 *
 * A replicate set or a replicated comparison is one job made of several
 * trainings. The server writes it a run of its own, with the mean, the interval
 * and the paired tests copied from the job's report, and tags each seed's run
 * with the group's id. This file decides how those fit on screen: which runs
 * fold under which group, how an aggregate is printed, and what the verdict
 * columns say.
 *
 * Nothing here computes a statistic. The one subtraction, the half-width of an
 * interval, only prints the report's own bounds in the `mean ± half` form the
 * results view already uses; every p-value, mean and `significant` flag is read.
 * Pure and free of text: the words come from the dictionary (`runGroup`).
 */
import type {
  GroupChild,
  MetricAggregate,
  PairedTest,
  RunGroup,
  RunSummary,
} from "../types/run";

/** A History row: a run, with the seeds folded under it when it is a group. */
export interface HistoryEntry {
  run: RunSummary;
  children: RunSummary[];
}

export function isGroupRun(run: { group?: unknown }): boolean {
  return run.group !== null && run.group !== undefined;
}

/**
 * The list as the researcher reads it: a group is one entry and its seeds sit
 * under it. Order is kept (the server sends newest first), and a group's seeds
 * follow the order the group recorded them in.
 *
 * A seed folds only under a group that is in the list. If the group was deleted
 * (that removes its summary and nothing else), the seeds it left behind carry a
 * `group_id` that points nowhere; they stay visible as the ordinary runs they
 * are rather than vanishing with the folder that named them.
 */
export function foldGroups(runs: ReadonlyArray<RunSummary>): HistoryEntry[] {
  const groups = new Map<string, RunSummary>();
  for (const run of runs) {
    if (isGroupRun(run)) groups.set(run.run_id, run);
  }
  const under = new Map<string, RunSummary[]>();
  const top: RunSummary[] = [];
  for (const run of runs) {
    const owner = run.group_id ? groups.get(run.group_id) : undefined;
    if (owner && !isGroupRun(run)) {
      under.set(owner.run_id, [...(under.get(owner.run_id) ?? []), run]);
    } else {
      top.push(run);
    }
  }
  return top.map((run) => {
    const children = under.get(run.run_id) ?? [];
    const order = run.group?.child_ids ?? [];
    const rank = (id: string) => {
      const at = order.indexOf(id);
      return at === -1 ? Number.MAX_SAFE_INTEGER : at;
    };
    return {
      run,
      children: [...children].sort((a, b) => rank(a.run_id) - rank(b.run_id)),
    };
  });
}

/** Half the width of the interval the report gave, for the `mean ± half` form.
 *  Null when the group has fewer than two seeds and so no interval. */
export function ciHalfWidth(
  agg: Pick<MetricAggregate, "ci95_low" | "ci95_high"> | null | undefined,
): number | null {
  if (!agg) return null;
  const { ci95_low: low, ci95_high: high } = agg;
  if (typeof low !== "number" || typeof high !== "number") return null;
  if (!Number.isFinite(low) || !Number.isFinite(high)) return null;
  return (high - low) / 2;
}

function fixed(value: number | null | undefined, digits: number): string | null {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : null;
}

/** A number for a cell: four decimals, a dash where there is none (a metric a
 *  seed never reported, or a value the server could not carry). */
export function formatNumber(value: number | null | undefined, digits = 4): string {
  return fixed(value, digits) ?? "—";
}

export interface FormattedAggregate {
  mean: string;
  /** Half-width of the interval; null when there is no interval. */
  half: string | null;
  low: string | null;
  high: string | null;
  std: string | null;
  n: number;
}

/** An aggregate as strings, or null when it has no mean to print. */
export function formatAggregate(
  agg: MetricAggregate | null | undefined,
  digits = 4,
): FormattedAggregate | null {
  if (!agg) return null;
  const mean = fixed(agg.mean, digits);
  if (mean === null) return null;
  return {
    mean,
    half: fixed(ciHalfWidth(agg), digits),
    low: fixed(agg.ci95_low, digits),
    high: fixed(agg.ci95_high, digits),
    std: fixed(agg.std, digits),
    n: agg.n,
  };
}

/** The group's aggregate for one row of a run.json `metrics` table, or null.
 *  The server says which aggregate filled which key (`metric_keys`), so a
 *  validation-score row is never given a test-split interval by name. */
export function aggregateForRow(
  group: Pick<RunGroup, "metric_keys" | "aggregates"> | null | undefined,
  rowKey: string,
): MetricAggregate | null {
  const name = group?.metric_keys?.[rowKey];
  if (!name) return null;
  return group?.aggregates?.[name] ?? null;
}

/** Why a replicated comparison names no best variant, when it names none. */
export type NoBestReason = "stopped" | "too-few-seeds";

export type BestByMean =
  | { kind: "best"; label: string }
  | { kind: "none"; reason: NoBestReason };

/**
 * The variant with the best mean, or the reason there is none.
 *
 * The server ranks only variants the paired tests could take, and only on the
 * seeds they all finished, so it can come back empty: a stop left variants
 * unrun, or fewer than two seeds are common to the variants that did run.
 * "Best by mean" is a description of the means, never a claim of significance;
 * that is what the paired tests are for.
 */
export function bestByMean(
  group: Pick<RunGroup, "best_by_mean" | "not_run">,
): BestByMean {
  if (group.best_by_mean) return { kind: "best", label: group.best_by_mean };
  return {
    kind: "none",
    reason: (group.not_run?.length ?? 0) > 0 ? "stopped" : "too-few-seeds",
  };
}

/** A p-value for a cell: four decimals, `<0.0001` below that, a dash for none. */
export function formatP(p: number | null | undefined): string {
  if (typeof p !== "number" || !Number.isFinite(p)) return "—";
  return p < 0.0001 ? "<0.0001" : p.toFixed(4);
}

/** The verdict column: the server's Holm-corrected flag, read as yes or no. */
export function holmVerdict(test: Pick<PairedTest, "significant">): "yes" | "no" {
  return test.significant ? "yes" : "no";
}

/** The run to open for a seed, or null when it has none or it is gone.
 *  `known` is the ids History lists: a seed deleted since the group was
 *  written would otherwise be a link to a 404. */
export function childTarget(
  child: Pick<GroupChild, "run_id">,
  known: ReadonlySet<string> | undefined,
): string | null {
  if (!child.run_id) return null;
  if (known && !known.has(child.run_id)) return null;
  return child.run_id;
}

/** Seeds shown in a few-seeds caution: below this the interval is a poor guide
 *  (core/significance.py says neither interval is trustworthy under ~5). */
export const FEW_SEEDS = 5;

export function hasFewSeeds(n: number): boolean {
  return n < FEW_SEEDS;
}
