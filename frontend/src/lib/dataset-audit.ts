import type { DatasetAuditLayout, DatasetAuditRequest } from "../api/client";

/** What a task panel tells the audit about its dataset (ADR-230). */
export interface DatasetAuditProps {
  baseDir: string;
  /** Split name → what the layout reads for it: a folder, a YOLO split or a CSV. */
  splits: Record<string, string>;
  /** How the task keeps its splits (ADR-230); folders by default. */
  layout?: DatasetAuditLayout;
  /** CSV layout only: the folder the CSV paths are relative to, and their column. */
  imagesDir?: string;
  imageColumn?: string;
  /** Splits training runs without; absent, they are no finding. */
  optionalSplits?: string[];
  /** One line on what this task's audit reads, under the scope. */
  note?: string;
}

/** The request for a task's layout; the CSV fields go only with the CSV layout. */
export function auditRequest(
  { baseDir, splits, layout = "folders", imagesDir, imageColumn, optionalSplits }: Omit<DatasetAuditProps, "note">,
  options: Pick<DatasetAuditRequest, "group_manifest" | "strict_groups" | "near_duplicates">,
): DatasetAuditRequest {
  return {
    base_dir: baseDir,
    // An optional split left blank (a regression test CSV) is not audited.
    splits: Object.fromEntries(Object.entries(splits).filter(([, value]) => value.trim())),
    layout,
    ...(layout === "csv" ? { images_dir: imagesDir || "images", image_column: imageColumn || "image" } : {}),
    ...(optionalSplits?.length ? { optional_splits: optionalSplits } : {}),
    ...options,
  };
}
