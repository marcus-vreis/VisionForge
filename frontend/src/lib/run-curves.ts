/**
 * The per-epoch charts of one run's detail: which series share a chart.
 *
 * A run's train and validation numbers of the same quantity (`train_loss` and
 * `val_loss`) are read against each other, so they share a chart. Only the
 * quantities the task charts first (lib/compare-curves.ts) get one: a detection
 * history has a dozen series, and a results page with a dozen charts answers
 * nothing at a glance. The rest stays in the comparison, where it is picked.
 *
 * Pure, and free of text: a series carries its dictionary key, as there.
 */
import { curveSeries, type CurveSeries } from "./compare-curves";

export type RunCurveRole = "train" | "val";

export interface RunCurve extends CurveSeries {
  /** Which split measured it; a name with no `train_` prefix is a validation number. */
  role: RunCurveRole;
}

export interface RunCurveGroup {
  /** The quantity: the history key without its split prefix. */
  id: string;
  /** Validation first, then train. */
  series: RunCurve[];
}

/** Train and validation lines, the same in the live tab and in the run detail. */
export const TRAIN_COLOR = "oklch(calc(0.74 - var(--lt-flip, 0) * 0.18) 0.16 240)";
export const VAL_COLOR = "oklch(calc(0.84 - var(--lt-flip, 0) * 0.38) 0.18 75)";

type History = ReadonlyArray<Record<string, unknown>>;

function roleOf(key: string): RunCurveRole {
  return key.startsWith("train_") ? "train" : "val";
}

/** `train_box_loss`, `val_box_loss` and detection's older `box_loss` are one quantity. */
export function curveQuantity(key: string): string {
  return key.replace(/^(?:train|val)_/, "");
}

/**
 * The charts of a run, in the order its task charts them. Empty when the run
 * kept no history (a replicate group) or measured none of the task's series.
 */
export function runCurveGroups(
  task: string,
  history: History,
  served?: Readonly<Record<string, string>>,
): RunCurveGroup[] {
  const series = curveSeries(task, [history], served);
  const groups: RunCurveGroup[] = [];
  for (const lead of series.filter((s) => s.initial)) {
    const id = curveQuantity(lead.key);
    if (groups.some((g) => g.id === id)) continue;
    const members = series
      .filter((s) => curveQuantity(s.key) === id)
      .map((s) => ({ ...s, role: roleOf(s.key) }));
    // One line per split: an alias of the same number (`box_loss`) is not drawn twice.
    const perRole = (role: RunCurveRole) => members.find((s) => s.role === role);
    groups.push({
      id,
      series: [perRole("val"), perRole("train")].filter((s): s is RunCurve => s !== undefined),
    });
  }
  return groups;
}

/** A history as chart points: an epoch that did not measure the series is a gap. */
export function curvePoints(history: History, key: string): { x: number; y: number | null }[] {
  return history.flatMap((record) => {
    const epoch = record["epoch"];
    if (typeof epoch !== "number" || !Number.isFinite(epoch)) return [];
    const value = record[key];
    return [{ x: epoch, y: typeof value === "number" && Number.isFinite(value) ? value : null }];
  });
}
