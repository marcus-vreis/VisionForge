/** Pure helpers of the dataset explorer (ADR-164): request shaping, paging,
 *  distribution bars and scan progress. Kept out of the component so they are
 *  unit-testable without a DOM. */

export type ExplorerTask = "classification" | "detection";
export type SplitName = "train" | "val" | "test";
export const SPLITS: SplitName[] = ["train", "val", "test"];

export interface ExplorerForm {
  task: ExplorerTask;
  base_dir: string;
  train_dir: string;
  val_dir: string;
  test_dir: string;
}

export function makeDefaultExplorerForm(): ExplorerForm {
  return {
    task: "classification",
    base_dir: "",
    train_dir: "train",
    val_dir: "val",
    test_dir: "test",
  };
}

/** The body every explorer call shares. The split folders only matter for an
 *  ImageFolder, so a detection request leaves them out. */
export function explorerBody(form: ExplorerForm): Record<string, string> {
  const body: Record<string, string> = {
    base_dir: form.base_dir.trim(),
    task: form.task,
  };
  if (form.task === "classification") {
    body.train_dir = form.train_dir || "train";
    body.val_dir = form.val_dir || "val";
    body.test_dir = form.test_dir || "test";
  }
  return body;
}

export const GALLERY_PAGE_SIZE = 48;

export interface Paging {
  page: number; // 0-based
  pages: number;
  from: number; // 1-based first item shown, 0 when empty
  to: number;
}

export function pagingOf(total: number, pageSize: number, offset: number): Paging {
  if (total <= 0) return { page: 0, pages: 0, from: 0, to: 0 };
  const page = Math.floor(offset / pageSize);
  return {
    page,
    pages: Math.ceil(total / pageSize),
    from: offset + 1,
    to: Math.min(total, offset + pageSize),
  };
}

/** Offset of the page `delta` away, clamped to the existing pages. */
export function stepOffset(
  total: number,
  pageSize: number,
  offset: number,
  delta: number,
): number {
  const last = Math.max(0, Math.ceil(total / pageSize) - 1);
  const page = Math.min(last, Math.max(0, Math.floor(offset / pageSize) + delta));
  return page * pageSize;
}

export interface SplitSummary {
  present: boolean;
  images: number;
  per_class: Record<string, number>;
  boxes: Record<string, number>;
  unlabeled: number;
}

export interface ExplorerSummary {
  base_dir: string;
  task: ExplorerTask;
  class_names: string[];
  total_images: number;
  splits: Record<SplitName, SplitSummary>;
}

export interface Bar {
  label: string;
  value: number;
  /** 0-100, relative to the largest bar of the same chart. */
  pct: number;
}

/** Bars for a count map, largest first, widths relative to the largest. */
export function barsOf(counts: Record<string, number>): Bar[] {
  const entries = Object.entries(counts).sort(
    (a, b) => b[1] - a[1] || a[0].localeCompare(b[0]),
  );
  const max = entries.reduce((m, [, v]) => Math.max(m, v), 0);
  return entries.map(([label, value]) => ({
    label,
    value,
    pct: max > 0 ? (value / max) * 100 : 0,
  }));
}

/** Class x split table rows: one row per class, one count per split. */
export function classSplitRows(
  summary: ExplorerSummary,
  source: "per_class" | "boxes",
): { name: string; counts: Record<SplitName, number>; total: number }[] {
  return summary.class_names.map((name) => {
    const counts = { train: 0, val: 0, test: 0 } as Record<SplitName, number>;
    for (const s of SPLITS) counts[s] = summary.splits[s][source][name] ?? 0;
    return { name, counts, total: counts.train + counts.val + counts.test };
  });
}

export interface ScanFile {
  split: string;
  path: string;
}

export interface DuplicateGroup {
  hash: string;
  splits: string[];
  classes: string[];
  files: ScanFile[];
  label_conflict: boolean;
}

export interface Bin {
  lo: number;
  hi: number | null;
  count: number;
}

export interface SizeReport {
  count: number;
  width?: { min: number; max: number; mean: number; median: number };
  height?: { min: number; max: number; mean: number; median: number };
  distinct_sizes?: number;
  top_sizes?: { width: number; height: number; count: number }[];
  side_bins?: Bin[];
  aspect_bins?: Bin[];
}

export interface ScanReport {
  total_images: number;
  scanned: number;
  limit: number;
  sampled: boolean;
  sample_step: number;
  cancelled: boolean;
  complete: boolean;
  corrupted_count: number;
  corrupted: { split: string; path: string; reason: string }[];
  duplicate_group_count: number;
  duplicate_file_count: number;
  duplicate_groups: DuplicateGroup[];
  leakage_group_count: number;
  leakage_groups: DuplicateGroup[];
  sizes: SizeReport;
}

export type ScanState = "running" | "done" | "cancelled" | "failed";

export interface ScanJob {
  job_id: string;
  state: ScanState;
  done: number;
  total: number;
  result: ScanReport | null;
  error: string | null;
}

export function isScanFinished(state: ScanState): boolean {
  return state !== "running";
}

/** Whole-number percentage of a scan, 0 until the total is known. */
export function scanPercent(done: number, total: number): number {
  if (total <= 0) return 0;
  return Math.min(100, Math.floor((done / total) * 100));
}

export type ProblemKind = "corrupted" | "duplicates" | "leakage";

/** How many findings of each kind, in the order the list shows them. */
export function problemCounts(report: ScanReport): Record<ProblemKind, number> {
  return {
    corrupted: report.corrupted_count,
    duplicates: report.duplicate_group_count,
    leakage: report.leakage_group_count,
  };
}

export function hasProblems(report: ScanReport): boolean {
  const c = problemCounts(report);
  return c.corrupted + c.duplicates + c.leakage > 0;
}

/** "64-128", "2048+" for a half-open bin. */
export function binLabel(bin: Bin, digits = 0): string {
  const fmt = (v: number) => (digits > 0 ? v.toFixed(digits) : String(v));
  return bin.hi === null ? `${fmt(bin.lo)}+` : `${fmt(bin.lo)}-${fmt(bin.hi)}`;
}

export function binBars(bins: Bin[], digits = 0): Bar[] {
  const max = bins.reduce((m, b) => Math.max(m, b.count), 0);
  return bins.map((b) => ({
    label: binLabel(b, digits),
    value: b.count,
    pct: max > 0 ? (b.count / max) * 100 : 0,
  }));
}

/** URL of a server-resized thumbnail. `root` and `path` are encoded, so a
 *  folder with `&` or `#` in its name still reaches the server intact. */
export function thumbUrl(
  root: string,
  path: string,
  size: number,
  labelPath: string | null,
): string {
  const q = new URLSearchParams({ root, path, size: String(size) });
  if (labelPath) q.set("label_path", labelPath);
  return `/api/explorer/thumb?${q.toString()}`;
}
