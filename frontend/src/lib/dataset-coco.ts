/** Pure helpers of "export to COCO / import COCO" (ADR-169): the shapes the API
 *  answers with, the split choices of the import and the counts shown. Kept out of
 *  the components so they are unit-testable without a DOM. */

import { SPLITS, type SplitName } from "./dataset-explorer";

export interface SplitCount {
  images: number;
  boxes: number;
}

/** An image the export leaves out (it does not open, or its label is malformed). */
export interface ExcludedExportImage {
  path: string;
  split: string;
  reason: string;
  detail?: string;
}

export interface CocoExportPreview {
  suggested_name: string;
  base_dir: string;
  classes: string[];
  categories: number;
  present: Record<SplitName, boolean>;
  images: number;
  boxes: number;
  per_split: Record<SplitName, SplitCount>;
  excluded: number;
  excluded_by_reason: Record<string, number>;
  excluded_listed: ExcludedExportImage[];
}

export interface CocoExportResult {
  path: string;
  images: number;
  boxes: number;
  excluded: number;
  files_linked: number;
  files_copied: number;
}

/** One thing left out or changed in a COCO file, as the API lists it. */
export interface CocoListedItem {
  json: string;
  item: string;
  reason: string;
}

export interface CocoImportSource {
  name: string;
  detected_split: SplitName | null;
  images_dir: string | null;
  images_listed: number;
  images_kept: number;
  annotations: number;
  boxes: number;
  categories: number;
  skipped_by_reason: Record<string, number>;
  adjusted_by_reason: Record<string, number>;
  listed: CocoListedItem[];
}

export interface CocoImportPreview {
  suggested_name: string;
  base_dir: string;
  sources: CocoImportSource[];
  ignored: { name: string; reason: string }[];
}

export type SourceStatus = "ready" | "unassigned" | "skipped";

export interface CocoClassMapping {
  coco_id: number;
  yolo_id: number;
  name: string;
  original_name: string;
}

export interface CocoImportPlan {
  ready: boolean;
  sources: {
    name: string;
    split: SplitName | null;
    detected_split: SplitName | null;
    status: SourceStatus;
  }[];
  unassigned: string[];
  per_split: Record<SplitName, SplitCount>;
  images: number;
  boxes: number;
  classes: string[];
  mapping: CocoClassMapping[];
  renamed: CocoClassMapping[];
}

export interface CocoImportResult {
  path: string;
  images: number;
  boxes: number;
  classes: number;
  files_linked: number;
  files_copied: number;
}

/** What the user picked for a JSON: a split, `"skip"`, or `""` for "not chosen yet". */
export type SplitChoice = SplitName | "skip" | "";

export const SKIP = "skip";

/** The starting choice of every JSON: the split its name says, or nothing. */
export function initialChoices(
  sources: CocoImportSource[],
): Record<string, SplitChoice> {
  const choices: Record<string, SplitChoice> = {};
  for (const s of sources) choices[s.name] = s.detected_split ?? "";
  return choices;
}

/** The `splits` of the plan and apply calls: only what the user has decided. */
export function splitsBody(
  choices: Record<string, SplitChoice>,
): Record<string, string> {
  const body: Record<string, string> = {};
  for (const [name, choice] of Object.entries(choices)) {
    if (choice !== "") body[name] = choice;
  }
  return body;
}

/** Splits that more than one JSON is assigned to (the server refuses that). */
export function splitConflicts(
  choices: Record<string, SplitChoice>,
): SplitName[] {
  const counts: Record<string, number> = {};
  for (const choice of Object.values(choices)) {
    if (choice !== "" && choice !== SKIP)
      counts[choice] = (counts[choice] ?? 0) + 1;
  }
  return SPLITS.filter((s) => (counts[s] ?? 0) > 1);
}

/** Files without a decision, which keep the import from starting. */
export function undecided(choices: Record<string, SplitChoice>): number {
  return Object.values(choices).filter((c) => c === "").length;
}

export interface ReasonRow {
  reason: string;
  count: number;
}

/** Reasons with their counts, the most frequent first (ties by name). */
export function reasonRows(counts: Record<string, number>): ReasonRow[] {
  return Object.entries(counts)
    .filter(([, count]) => count > 0)
    .map(([reason, count]) => ({ reason, count }))
    .sort((a, b) => b.count - a.count || a.reason.localeCompare(b.reason));
}

export function reasonTotal(counts: Record<string, number>): number {
  return Object.values(counts).reduce((sum, n) => sum + n, 0);
}

/** Whether the import can start: a ready plan, a usable name, nothing running. */
export function canApplyImport(
  plan: CocoImportPlan | null,
  destOk: boolean,
  busy: boolean,
): boolean {
  return plan !== null && plan.ready && plan.images > 0 && destOk && !busy;
}

/** The sources that will feed the result (ready), for the summary lines. */
export function readySources(plan: CocoImportPlan): CocoImportPlan["sources"] {
  return plan.sources.filter((s) => s.status === "ready");
}
