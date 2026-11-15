/** The end-of-job report of a replicated comparison (ADR-061), read as the
 *  History group it becomes (ADR-113), so the results view and the group's
 *  detail draw it with the same component. A port of
 *  `core/run_groups._copy_comparison_group`: the frontend computes no
 *  statistics, it only renames what the report already holds. */

import type { GroupChild, GroupVariant, MetricAggregate, PairedTest, RunGroup } from "../types/run";

type Dict = Record<string, unknown>;

const asDict = (value: unknown): Dict =>
  value !== null && typeof value === "object" && !Array.isArray(value) ? (value as Dict) : {};

const asList = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

const asStrings = (value: unknown): string[] =>
  asList(value).filter((v): v is string => typeof v === "string");

const asNumbers = (value: unknown): number[] =>
  asList(value).filter((v): v is number => typeof v === "number");

/** The report `/api/experiment/result` returns for a replicated comparison. */
export function isReplicatedComparisonReport(report: Dict): boolean {
  return report["kind"] === "replicated_comparison" && typeof report["variants"] === "object";
}

/** One seed of a variant, as History lists it: its run is the folder it trained in. */
function childOf(row: unknown, variant: string): GroupChild {
  const r = asDict(row);
  const runDir = typeof r["run_dir"] === "string" ? r["run_dir"] : "";
  const folder = runDir.split(/[\\/]/).filter(Boolean).pop() ?? "";
  return {
    seed: typeof r["seed"] === "number" ? r["seed"] : null,
    variant,
    status: typeof r["status"] === "string" ? r["status"] : "failed",
    run_id: folder || null,
    run_dir: runDir || null,
    metrics: asDict(r["metrics"]) as Record<string, number | null>,
    error: typeof r["error"] === "string" ? r["error"] : "",
  };
}

/** The report as the `group` section its History entry carries. */
export function comparisonGroupFromReport(report: Dict): RunGroup {
  const seeds = asNumbers(report["seeds"]);
  const variants: Record<string, GroupVariant> = {};
  let finished = 0;
  for (const [label, raw] of Object.entries(asDict(report["variants"]))) {
    const v = asDict(raw);
    const children = asList(v["trials"]).map((row) => childOf(row, label));
    const done = children.filter((c) => c.status === "success");
    finished += done.length;
    variants[label] = {
      overrides: asDict(v["overrides"]),
      aggregates: asDict(v["aggregates"]) as Record<string, MetricAggregate>,
      successful: typeof v["successful"] === "number" ? v["successful"] : null,
      seeds_finished: done.map((c) => c.seed).filter((s): s is number => s !== null),
      children,
    };
  }
  // Trainings, not seeds: the variants a stop kept from starting count too.
  const notRun = asStrings(report["not_run"]);
  const direction = report["metric_direction"];
  return {
    kind: "replicated_comparison",
    metric: typeof report["metric"] === "string" ? report["metric"] : null,
    seeds,
    n_requested: (Object.keys(variants).length + notRun.length) * seeds.length,
    n_finished: finished,
    stopped: report["stopped"] === true,
    std_ddof: 1,
    report_path: null,
    report_dir: typeof report["report_dir"] === "string" ? report["report_dir"] : null,
    metric_direction: direction === "higher" || direction === "lower" ? direction : null,
    alpha: typeof report["alpha"] === "number" ? report["alpha"] : null,
    variants,
    comparisons: asList(report["comparisons"]) as PairedTest[],
    best_by_mean: typeof report["best_by_mean"] === "string" ? report["best_by_mean"] : null,
    ranked_by_mean: asStrings(report["ranked_by_mean"]),
    ranking_seeds: asNumbers(report["ranking_seeds"]),
    significant_pairs:
      typeof report["significant_pairs"] === "number" ? report["significant_pairs"] : null,
    skipped_variants: asStrings(report["skipped_variants"]),
    not_run: notRun,
    underpowered: report["underpowered"] === true,
  };
}
