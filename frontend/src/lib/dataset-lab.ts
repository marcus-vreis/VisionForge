/** Pure helpers of the dataset lab: the full-screen surface that opens a chosen
 *  dataset with its overview and its tools, one on screen at a time. Kept out
 *  of the components so they are unit-testable without a DOM. */

import { SPLITS, type ExplorerSummary, type ExplorerTask, type SplitName } from "./dataset-explorer";

export type LabSection = "overview" | "resplit" | "incorporate" | "merge" | "coco";

/** Every section, in the order of the rail. */
export const LAB_SECTIONS: readonly LabSection[] = [
  "overview",
  "resplit",
  "incorporate",
  "merge",
  "coco",
];

export type CocoMode = "export" | "import";

/** The sections the lab offers. With a dataset open, all of them; without one
 *  (the lab opened only to convert a COCO folder), only COCO, whose import does
 *  not need an opened dataset. */
export function labSections(task: ExplorerTask | null): LabSection[] {
  return task === null ? ["coco"] : [...LAB_SECTIONS];
}

/** What the COCO section can do. Export reads the opened YOLO dataset, so it is
 *  only offered for detection; import reads a folder the user picks and always is. */
export function cocoModes(task: ExplorerTask | null): CocoMode[] {
  return task === "detection" ? ["export", "import"] : ["import"];
}

/** The section to show first: the wanted one when offered, else the first offered. */
export function initialSection(task: ExplorerTask | null, wanted?: LabSection): LabSection {
  const offered = labSections(task);
  return wanted && offered.includes(wanted) ? wanted : offered[0];
}

/** The section a key moves to in the rail (arrow keys, Home, End; wraps), or
 *  null when the key does not move. */
export function stepSection(
  sections: readonly LabSection[],
  current: LabSection,
  key: string,
): LabSection | null {
  if (sections.length === 0) return null;
  const i = Math.max(0, sections.indexOf(current));
  const n = sections.length;
  switch (key) {
    case "ArrowDown":
    case "ArrowRight":
      return sections[(i + 1) % n];
    case "ArrowUp":
    case "ArrowLeft":
      return sections[(i - 1 + n) % n];
    case "Home":
      return sections[0];
    case "End":
      return sections[n - 1];
    default:
      return null;
  }
}

/** Focus trap: where Tab must jump to keep the focus inside the dialog.
 *  `index` is the focused element among the `count` focusable ones (-1 when the
 *  focus is outside). Null means the browser's own move is already inside. */
export function trapTarget(index: number, count: number, backwards: boolean): number | null {
  if (count <= 0) return null;
  if (index < 0) return backwards ? count - 1 : 0;
  if (backwards && index === 0) return count - 1;
  if (!backwards && index >= count - 1) return 0;
  return null;
}

/** Shortens a long path by its middle, so both the drive and the folder name
 *  stay readable: "C:/datasets/…/cafe/2024". */
export function middleTruncate(text: string, max: number): string {
  if (max < 3 || text.length <= max) return text;
  const keep = max - 1;
  const head = Math.ceil(keep / 2);
  const tail = keep - head;
  return `${text.slice(0, head)}…${tail > 0 ? text.slice(text.length - tail) : ""}`;
}

/** The dataset's name: the last folder of its path, whatever the separator. */
export function datasetName(path: string): string {
  const parts = path.trim().split(/[\\/]+/).filter((p) => p !== "");
  return parts.length > 0 ? parts[parts.length - 1] : path.trim();
}

export interface ClassStack {
  name: string;
  counts: Record<SplitName, number>;
  total: number;
  /** Width of each split's segment, 0-100 of the largest class total. */
  pct: Record<SplitName, number>;
}

/** One stacked bar per class (its images in each split), largest class first.
 *  Widths are relative to the largest class, so the bars compare with each other. */
export function classStacks(
  summary: ExplorerSummary,
  source: "per_class" | "boxes" = "per_class",
): ClassStack[] {
  const rows = summary.class_names.map((name) => {
    const counts = { train: 0, val: 0, test: 0 } as Record<SplitName, number>;
    for (const s of SPLITS) counts[s] = summary.splits[s][source][name] ?? 0;
    return { name, counts, total: counts.train + counts.val + counts.test };
  });
  const max = rows.reduce((m, r) => Math.max(m, r.total), 0);
  return rows
    .sort((a, b) => b.total - a.total || a.name.localeCompare(b.name))
    .map((r) => {
      const pct = { train: 0, val: 0, test: 0 } as Record<SplitName, number>;
      for (const s of SPLITS) pct[s] = max > 0 ? (r.counts[s] / max) * 100 : 0;
      return { ...r, pct };
    });
}

export type SummaryNote =
  | { kind: "empty" }
  | { kind: "missing_split"; split: SplitName }
  | { kind: "unlabeled"; count: number };

/** What the compact summary must warn about before the lab opens: no images at
 *  all, a missing train or validation folder, detection images without boxes.
 *  A missing test folder is common and is shown in the split counts instead. */
export function summaryNotes(summary: ExplorerSummary): SummaryNote[] {
  if (summary.total_images === 0) return [{ kind: "empty" }];
  const notes: SummaryNote[] = [];
  for (const s of ["train", "val"] as const) {
    if (!summary.splits[s].present) notes.push({ kind: "missing_split", split: s });
  }
  if (summary.task === "detection") {
    const unlabeled = SPLITS.reduce(
      (n, s) => n + (summary.splits[s].present ? summary.splits[s].unlabeled : 0),
      0,
    );
    if (unlabeled > 0) notes.push({ kind: "unlabeled", count: unlabeled });
  }
  return notes;
}

/** The lab opens only on a dataset with images, as the tools did before it. */
export function canOpenLab(summary: ExplorerSummary | null): boolean {
  return summary !== null && summary.total_images > 0;
}

/** Total boxes of a detection dataset, over the present splits. */
export function totalBoxes(summary: ExplorerSummary): number {
  return SPLITS.reduce(
    (n, s) =>
      n +
      Object.values(summary.splits[s].boxes).reduce((m, v) => m + v, 0),
    0,
  );
}
