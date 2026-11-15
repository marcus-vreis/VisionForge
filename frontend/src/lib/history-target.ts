/** Where a finished job lives in History, read from the result the server sent.
 *
 * History lists every run by the name of its folder (`RunSummary.run_id` is
 * `run_dir.name` in `run_history.py`), which is not the queue's job id:
 *
 * - One training (classification, transfer learning, regression, segmentation,
 *   anomaly, a Colab run brought back) names its folder under `report.train`;
 *   detection under `report.detection`; a researcher's task (ADR-058) at the top
 *   of its report.
 * - A job of the strategies (replicates, replicated comparison, a standalone
 *   task's K-fold, sweep or model comparison, any task's) writes its group under
 *   the job id itself (`record_run_group(group_id=run_id)`); its report is the
 *   one that carries `report_dir`.
 * - Classification's own K-fold, grid/random search and model comparison blocks
 *   name their group with a stamp taken when it is written, which the result
 *   does not carry. Their units' folders are in the report, and History tags
 *   each unit with its group's id, so the group is found through a unit.
 *
 * Anything else has no History entry this code can name, and the caller keeps
 * the results view.
 */

import type { RunResult, RunSummary } from "../types/run";

/** A History entry by its id, or the group one of these unit runs belongs to. */
export type HistoryTarget = { runId: string } | { childRunIds: string[] };

type Dict = Record<string, unknown>;

function asDict(value: unknown): Dict | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Dict)
    : null;
}

/** The last segment of a folder path, Windows or POSIX; null when there is none. */
export function folderName(path: unknown): string | null {
  if (typeof path !== "string") return null;
  const name = path.split(/[\\/]/).filter(Boolean).pop();
  return name ? name : null;
}

/** The folder names of the rows' `run_dir`, in order, without repeats. */
function unitFolders(rows: unknown): string[] {
  if (!Array.isArray(rows)) return [];
  const names: string[] = [];
  for (const row of rows) {
    const name = folderName(asDict(row)?.["run_dir"]);
    if (name && !names.includes(name)) names.push(name);
  }
  return names;
}

/** Where `result` is in History, or null when no entry can be named. */
export function historyTargetForResult(
  result: Pick<RunResult, "run_id" | "report">,
): HistoryTarget | null {
  const report = asDict(result.report);
  if (!report) return null;

  // A strategies job: its group is written under the job id.
  if (typeof report["report_dir"] === "string") return { runId: result.run_id };

  for (const section of ["train", "detection"]) {
    const name = folderName(asDict(report[section])?.["run_dir"]);
    if (name) return { runId: name };
  }
  // A researcher's task reports its folder at the top.
  if (typeof report["task"] === "string" && report["task"].startsWith("custom:")) {
    const name = folderName(report["run_dir"]);
    if (name) return { runId: name };
  }

  // Classification blocks that group their units under a stamped id.
  const units = [
    ...unitFolders(report["fold_results"]),
    ...unitFolders(report["top_3"]),
    ...unitFolders(asDict(report["best_trial"]) ? [report["best_trial"]] : []),
  ];
  return units.length > 0 ? { childRunIds: units } : null;
}

/** The History id `target` names among `runs`, or null while it is not listed.
 *
 * A unit opens its group, never itself: a K-fold's first fold is not the job
 * that finished. A unit with no group (its writer failed) names nothing.
 */
export function resolveHistoryTarget(
  target: HistoryTarget,
  runs: readonly Pick<RunSummary, "run_id" | "group_id">[],
): string | null {
  const listed = (id: string) => runs.some((r) => r.run_id === id);
  if ("runId" in target) return listed(target.runId) ? target.runId : null;
  for (const child of target.childRunIds) {
    const groupId = runs.find((r) => r.run_id === child)?.group_id;
    if (groupId && listed(groupId)) return groupId;
  }
  return null;
}
