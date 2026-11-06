/** Pure helpers of "incorporate images" (ADR-166): the preview and plan shapes,
 *  ratio and seed parsing, the destination-name rule and the numbers shown.
 *  Kept out of the component so they are unit-testable without a DOM. */

import { SPLITS, type SplitName } from "./dataset-explorer";

export type JobState = "running" | "done" | "cancelled" | "failed";

export interface IncorporateJob<R = Record<string, unknown>> {
  job_id: string;
  kind: "preview" | "apply";
  state: JobState;
  done: number;
  total: number;
  result: R | null;
  error: string | null;
}

export interface NewClass {
  name: string;
  images: number;
  source_id: number | null;
}

export interface IncorporatePreview {
  suggested_name: string;
  task: "classification" | "detection";
  base_dir: string;
  source_dir: string;
  base_classes: string[];
  base_counts: Record<SplitName, number>;
  present: Record<SplitName, boolean>;
  source_images: number;
  new_classes: NewClass[];
}

export interface ExcludedImage {
  path: string;
  reason: string;
  detail: string | null;
}

export interface IncorporatePlan {
  seed: number;
  ratios: Record<SplitName, number>;
  accepted_classes: string[];
  classes: string[];
  added: number;
  added_per_split: Record<SplitName, number>;
  added_per_class: Record<SplitName, Record<string, number>>;
  excluded_count: number;
  excluded_by_reason: Record<string, number>;
  excluded: ExcludedImage[];
}

export interface IncorporateResult {
  path: string;
  added: number;
  files_linked: number;
  files_copied: number;
  classes: string[];
}

/** The body of the plan and apply calls. `ratios` is omitted for "same as the dataset". */
export interface PlanBody {
  accept_classes: string[];
  seed: number;
  ratios?: Record<string, number>;
}

export function isJobFinished(state: JobState): boolean {
  return state !== "running";
}

export function jobPercent(done: number, total: number): number {
  if (total <= 0) return 0;
  return Math.max(0, Math.min(100, Math.round((done / total) * 100)));
}

/** Add or remove a class from the accepted list, keeping the order stable. */
export function toggleClass(accepted: string[], name: string): string[] {
  return accepted.includes(name)
    ? accepted.filter((n) => n !== name)
    : [...accepted, name].sort();
}

/** A non-negative integer seed from the text box; `null` when it is not one. */
export function parseSeed(text: string): number | null {
  const trimmed = text.trim();
  if (!/^\d{1,10}$/.test(trimmed)) return null;
  const n = Number(trimmed);
  return n <= 2 ** 31 - 1 ? n : null;
}

/** A non-negative percentage from a text box; empty counts as 0. */
export function parsePercent(text: string): number | null {
  const trimmed = text.trim().replace(",", ".");
  if (trimmed === "") return 0;
  if (!/^\d+(\.\d+)?$/.test(trimmed)) return null;
  const n = Number(trimmed);
  return Number.isFinite(n) ? n : null;
}

export type PercentTexts = Record<SplitName, string>;

/** Ratios typed by the user for the splits that exist. `null` when any box is
 *  not a number or nothing is above zero. Splits the dataset lacks are sent as 0. */
export function customRatios(
  texts: PercentTexts,
  present: Record<SplitName, boolean>,
): Record<string, number> | null {
  const out: Record<string, number> = {};
  let total = 0;
  for (const s of SPLITS) {
    const value = present[s] ? parsePercent(texts[s]) : 0;
    if (value === null) return null;
    out[s] = value;
    total += value;
  }
  return total > 0 ? out : null;
}

/** The request body, or `null` while the seed or the custom ratios are invalid. */
export function buildPlanBody(opts: {
  accepted: string[];
  seedText: string;
  custom: boolean;
  percents: PercentTexts;
  present: Record<SplitName, boolean>;
}): PlanBody | null {
  const seed = parseSeed(opts.seedText);
  if (seed === null) return null;
  const body: PlanBody = { accept_classes: opts.accepted, seed };
  if (opts.custom) {
    const ratios = customRatios(opts.percents, opts.present);
    if (ratios === null) return null;
    body.ratios = ratios;
  }
  return body;
}

/** Mirrors `validate_path_component` of the backend: one folder name, nothing the
 *  file systems reject. The server stays the authority; this only avoids a round trip. */
export function isValidDestName(name: string): boolean {
  if (name.trim() === "" || name === "." || name === "..") return false;
  if (name.endsWith(".") || name.endsWith(" ")) return false;
  if (/[/\\<>:"|?*]/.test(name)) return false;
  // eslint-disable-next-line no-control-regex
  if (/[\u0000-\u001f\u007f]/.test(name)) return false;
  const stem = name.split(".")[0].replace(/ +$/, "").toUpperCase();
  return !/^(CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|(COM|LPT)[1-9])$/.test(stem);
}

/** Ratios as "70 / 20 / 10" over the splits that exist. */
export function ratioLabel(
  ratios: Record<SplitName, number>,
  present: Record<SplitName, boolean>,
): string {
  return SPLITS.filter((s) => present[s])
    .map((s) => String(Math.round(ratios[s] * 100)))
    .join(" / ");
}

export interface PlanRow {
  name: string;
  counts: Record<SplitName, number>;
  total: number;
}

/** One row per class that receives images, one count per split. */
export function planRows(plan: IncorporatePlan): PlanRow[] {
  const names = new Set<string>();
  for (const s of SPLITS) for (const n of Object.keys(plan.added_per_class[s])) names.add(n);
  return [...names].sort().map((name) => {
    const counts = { train: 0, val: 0, test: 0 } as Record<SplitName, number>;
    for (const s of SPLITS) counts[s] = plan.added_per_class[s][name] ?? 0;
    return { name, counts, total: counts.train + counts.val + counts.test };
  });
}

/** Reasons for leaving images out, most frequent first. */
export function reasonRows(plan: IncorporatePlan): { reason: string; count: number }[] {
  return Object.entries(plan.excluded_by_reason)
    .map(([reason, count]) => ({ reason, count }))
    .sort((a, b) => b.count - a.count || a.reason.localeCompare(b.reason));
}

/** The form that opens the new dataset in the explorer: same task and split names. */
export function openResultForm<F extends { base_dir: string }>(form: F, path: string): F {
  return { ...form, base_dir: path };
}
