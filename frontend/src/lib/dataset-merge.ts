/** Pure helpers of "merge with another dataset" (ADR-167): the preview and plan
 *  shapes, the class-mapping table (one row per class of A and of B), the target
 *  checks and the numbers shown. Kept out of the component so they are
 *  unit-testable without a DOM. */

import { SPLITS, type SplitName } from "./dataset-explorer";
import { isValidDestName } from "./dataset-incorporate";

export type Side = "a" | "b";
export const SIDES: Side[] = ["a", "b"];

export interface MergeClass {
  name: string;
  images: number;
}

export interface MergePreview {
  suggested_name: string;
  task: "classification" | "detection";
  a_dir: string;
  b_dir: string;
  classes: Record<Side, MergeClass[]>;
  present: Record<Side, Record<SplitName, boolean>>;
  found: Record<Side, number>;
  repeated_within: Record<Side, number>;
  /** Groups of names that differ only by case, spaces, `_` or `-`. */
  similar_names: string[][];
}

export interface MergeDropped {
  side: Side;
  path: string;
  reason: string;
}

export interface MergeDuplicate {
  b_path: string;
  a_path: string;
  b_split: SplitName;
  a_split: SplitName;
  cross_split: boolean;
}

export interface MergePlanClass {
  name: string;
  images: Record<SplitName | "total", number>;
  boxes: number;
}

export interface MergePlanSplit {
  present: boolean;
  images: number;
  from_a: number;
  from_b: number;
}

export interface MergePlan {
  classes: MergePlanClass[];
  background: Record<SplitName, number>;
  splits: Record<SplitName, MergePlanSplit>;
  only_in: Record<Side, SplitName[]>;
  images_total: number;
  dropped_count: number;
  dropped_by_reason: Record<string, number>;
  dropped: MergeDropped[];
  duplicates_dropped: number;
  duplicates_cross_split: number;
  duplicates: MergeDuplicate[];
}

export interface MergeResult {
  path: string;
  images: number;
  files_linked: number;
  files_copied: number;
  classes: string[];
}

/** `{a: {class: target or null}, b: {...}}`; `null` discards the class. */
export type MappingBody = Record<Side, Record<string, string | null>>;

/** One row of the mapping table. `target` is what the box holds, `discard` wins over it. */
export interface MergeRow {
  side: Side;
  name: string;
  images: number;
  target: string;
  discard: boolean;
}

export type TargetProblem = "empty" | "invalid" | "case";

/** The table the page starts from: every class keeps its own name. */
export function initialRows(preview: MergePreview): MergeRow[] {
  return SIDES.flatMap((side) =>
    preview.classes[side].map((c) => ({
      side,
      name: c.name,
      images: c.images,
      target: c.name,
      discard: false,
    })),
  );
}

/** A copy of `rows` with the row `(side, name)` changed. */
export function updateRow(
  rows: MergeRow[],
  side: Side,
  name: string,
  patch: Partial<Pick<MergeRow, "target" | "discard">>,
): MergeRow[] {
  return rows.map((r) => (r.side === side && r.name === name ? { ...r, ...patch } : r));
}

/** The request body: every class listed, a discarded one as `null`. */
export function buildMapping(rows: MergeRow[]): MappingBody {
  const body: MappingBody = { a: {}, b: {} };
  for (const r of rows) body[r.side][r.name] = r.discard ? null : r.target.trim();
  return body;
}

/** Why a row's target cannot be used, or `null`. Mirrors the server's rules
 *  (it stays the authority): non-empty after trimming, a valid folder name for
 *  classification, no newline for detection. */
export function targetProblem(
  row: MergeRow,
  task: "classification" | "detection",
): TargetProblem | null {
  if (row.discard) return null;
  const target = row.target.trim();
  if (target === "") return "empty";
  if (task === "classification") return isValidDestName(target) ? null : "invalid";
  return /[\r\n]/.test(target) ? "invalid" : null;
}

/** Problems per row key (`side:name`). For classification, targets that differ
 *  only by letter case are a problem too: they would be one folder on Windows. */
export function rowProblems(
  rows: MergeRow[],
  task: "classification" | "detection",
): Map<string, TargetProblem> {
  const problems = new Map<string, TargetProblem>();
  const byFold = new Map<string, Set<string>>();
  for (const r of rows) {
    const key = rowKey(r);
    const own = targetProblem(r, task);
    if (own) {
      problems.set(key, own);
      continue;
    }
    if (r.discard) continue;
    const target = r.target.trim();
    const fold = target.toLowerCase();
    byFold.set(fold, (byFold.get(fold) ?? new Set()).add(target));
  }
  if (task === "classification") {
    for (const r of rows) {
      if (r.discard || problems.has(rowKey(r))) continue;
      if ((byFold.get(r.target.trim().toLowerCase())?.size ?? 0) > 1) problems.set(rowKey(r), "case");
    }
  }
  return problems;
}

export function rowKey(row: Pick<MergeRow, "side" | "name">): string {
  return `${row.side}:${row.name}`;
}

/** Names for the target datalist: the classes of both datasets, sorted, once each. */
export function targetSuggestions(preview: MergePreview): string[] {
  const names = new Set<string>();
  for (const side of SIDES) for (const c of preview.classes[side]) names.add(c.name);
  return [...names].sort((x, y) => x.localeCompare(y));
}

/** The kept targets in the order the result has them: A's first, then B's new ones. */
export function finalOrder(rows: MergeRow[]): string[] {
  const seen = new Set<string>();
  const order: string[] = [];
  for (const side of SIDES) {
    for (const r of rows) {
      if (r.side !== side || r.discard) continue;
      const target = r.target.trim();
      if (target !== "" && !seen.has(target)) {
        seen.add(target);
        order.push(target);
      }
    }
  }
  return order;
}

/** True when two rows end up with the same target: the table shows them as merged. */
export function mergedTargets(rows: MergeRow[]): Set<string> {
  const count = new Map<string, number>();
  for (const r of rows) {
    if (r.discard) continue;
    const target = r.target.trim();
    count.set(target, (count.get(target) ?? 0) + 1);
  }
  return new Set([...count].filter(([, n]) => n > 1).map(([target]) => target));
}

/** Similar-name groups that still sit in the table under different targets. */
export function activeHints(preview: MergePreview, rows: MergeRow[]): string[][] {
  const targets = new Set(rows.filter((r) => !r.discard).map((r) => r.target.trim()));
  return preview.similar_names.filter((group) => group.filter((n) => targets.has(n)).length > 1);
}

/** Splits that come from one dataset only, as `[side, split]` pairs. */
export function splitsFromOne(plan: MergePlan): [Side, SplitName][] {
  return SIDES.flatMap((side) => plan.only_in[side].map((s) => [side, s] as [Side, SplitName]));
}

/** Reasons for leaving images out, most frequent first. */
export function reasonRows(plan: MergePlan): { reason: string; count: number }[] {
  return Object.entries(plan.dropped_by_reason)
    .map(([reason, count]) => ({ reason, count }))
    .sort((a, b) => b.count - a.count || a.reason.localeCompare(b.reason));
}

/** Splits the result has, in the usual order. */
export function presentSplits(plan: MergePlan): SplitName[] {
  return SPLITS.filter((s) => plan.splits[s].present);
}

/** Whether the table can be sent: no row has a problem. */
export function canMerge(problems: Map<string, TargetProblem>, plan: MergePlan | null): boolean {
  return problems.size === 0 && plan !== null && plan.images_total > 0;
}
