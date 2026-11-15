/** Pure helpers of "re-split a dataset" (ADR-168): the preview and plan shapes, the
 *  request body, the before/after tables and the numbers shown. Kept out of the
 *  component so they are unit-testable without a DOM. Job states, seed and
 *  destination-name rules are shared with the incorporate panel. */

import { SPLITS, type SplitName } from "./dataset-explorer";
import { parsePercent, parseSeed, type PercentTexts } from "./dataset-incorporate";

export interface ResplitPreview {
  suggested_name: string;
  task: "classification" | "detection";
  base_dir: string;
  classes: string[];
  present: Record<SplitName, boolean>;
  images: number;
  per_split: Record<SplitName, number>;
  group_csv: string | null;
  group_count: number | null;
  ungrouped: number;
  unknown_entries: number;
}

export interface ResplitWarning {
  code: string;
  split?: SplitName;
  class?: string;
}

export interface DroppedImage {
  path: string;
  split: string;
  reason: string;
}

export interface ResplitPlan {
  seed: number;
  ratios: Record<SplitName, number>;
  achieved: Partial<Record<SplitName, number>>;
  locked_splits: SplitName[];
  by_group: boolean;
  before: Record<SplitName, Record<string, number>>;
  after: Record<SplitName, Record<string, number>>;
  before_totals: Record<SplitName, number>;
  after_totals: Record<SplitName, number>;
  kept: number;
  moved: number;
  dropped_count: number;
  dropped_by_reason: Record<string, number>;
  dropped: DroppedImage[];
  duplicate_sets: number;
  duplicate_extras: number;
  duplicate_sets_split_before: number;
  warnings: ResplitWarning[];
}

export interface ResplitResult {
  path: string;
  images: number;
  moved: number;
  files_linked: number;
  files_copied: number;
}

/** The body of the plan and apply calls. `ratios` is omitted for "keep the current
 *  proportion"; `by_group` is omitted when the user did not decide. */
export interface ResplitBody {
  seed: number;
  lock_test: boolean;
  ratios?: Record<string, number>;
  by_group?: boolean;
}

/** The test split is locked only if the user wants it and the dataset has one. */
export function effectiveLock(lockTest: boolean, present: Record<SplitName, boolean>): boolean {
  return lockTest && present.test;
}

/** Ratios typed by the user. A locked test is sent as 0; `null` when a box is not a
 *  number or nothing is above zero. */
export function customRatios(
  texts: PercentTexts,
  locked: boolean,
): Record<string, number> | null {
  const out: Record<string, number> = {};
  let total = 0;
  for (const s of SPLITS) {
    const value = locked && s === "test" ? 0 : parsePercent(texts[s]);
    if (value === null) return null;
    out[s] = value;
    total += value;
  }
  return total > 0 ? out : null;
}

/** The request body, or `null` while the seed or the custom ratios are invalid. */
export function buildBody(opts: {
  seedText: string;
  lockTest: boolean;
  present: Record<SplitName, boolean>;
  custom: boolean;
  percents: PercentTexts;
  /** `null`: no group file was loaded, so there is nothing to decide. */
  byGroup: boolean | null;
}): ResplitBody | null {
  const seed = parseSeed(opts.seedText);
  if (seed === null) return null;
  const locked = effectiveLock(opts.lockTest, opts.present);
  const body: ResplitBody = { seed, lock_test: opts.lockTest };
  if (opts.byGroup !== null) body.by_group = opts.byGroup;
  if (opts.custom) {
    const ratios = customRatios(opts.percents, locked);
    if (ratios === null) return null;
    body.ratios = ratios;
  }
  return body;
}

/** The pooled splits' own proportion, as whole percents, to start the boxes from. */
export function currentPercents(
  perSplit: Record<SplitName, number>,
  locked: boolean,
): PercentTexts {
  const pooled = SPLITS.filter((s) => !(locked && s === "test"));
  const total = pooled.reduce((n, s) => n + perSplit[s], 0) || 1;
  const pct = (s: SplitName) =>
    pooled.includes(s) ? String(Math.round((perSplit[s] / total) * 100)) : "0";
  return { train: pct("train"), val: pct("val"), test: pct("test") };
}

/** A share as a percentage with at most one decimal: 0.6 -> "60", 0.3333 -> "33.3". */
export function formatShare(share: number): string {
  return String(Math.round(share * 1000) / 10);
}

export interface RatioRow {
  split: SplitName;
  requested: number;
  achieved: number;
}

/** Requested against achieved share for the splits the re-split distributes over. */
export function ratioRows(plan: ResplitPlan): RatioRow[] {
  return SPLITS.filter((s) => !plan.locked_splits.includes(s))
    .map((split) => ({
      split,
      requested: plan.ratios[split] ?? 0,
      achieved: plan.achieved[split] ?? 0,
    }))
    .filter((r) => r.requested > 0 || r.achieved > 0);
}

export interface CompareRow {
  name: string;
  before: Record<SplitName, number>;
  after: Record<SplitName, number>;
}

/** One row per class: its images in each split before and after. */
export function compareRows(plan: ResplitPlan): CompareRow[] {
  const names = new Set<string>();
  for (const s of SPLITS) {
    for (const n of Object.keys(plan.before[s])) names.add(n);
    for (const n of Object.keys(plan.after[s])) names.add(n);
  }
  return [...names].sort().map((name) => {
    const before = { train: 0, val: 0, test: 0 } as Record<SplitName, number>;
    const after = { train: 0, val: 0, test: 0 } as Record<SplitName, number>;
    for (const s of SPLITS) {
      before[s] = plan.before[s][name] ?? 0;
      after[s] = plan.after[s][name] ?? 0;
    }
    return { name, before, after };
  });
}

/** Reasons for dropping images, most frequent first. */
export function droppedReasonRows(plan: ResplitPlan): { reason: string; count: number }[] {
  return Object.entries(plan.dropped_by_reason)
    .map(([reason, count]) => ({ reason, count }))
    .sort((a, b) => b.count - a.count || a.reason.localeCompare(b.reason));
}

/** Splits that exist in the dataset and have no images after the re-split. */
export function vanishedSplits(
  plan: ResplitPlan,
  present: Record<SplitName, boolean>,
): SplitName[] {
  return SPLITS.filter((s) => present[s] && plan.after_totals[s] === 0);
}
