/** The target analysis of a regression dataset (ADR-235). */
import type { AnalysisJob, SavedAnalysis } from "./analysis-job";

export interface TargetRequest {
  base_dir: string;
  images_dir: string;
  train_csv: string;
  val_csv: string;
  test_csv: string;
  image_column: string;
  target_columns: string[];
}

export interface TargetSplit {
  count: number;
  /** Blank or non-numeric cells. */
  bad: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  q1: number | null;
  median: number | null;
  q3: number | null;
  max: number | null;
  /** Counts on the target's shared bins over `range`. */
  histogram: number[];
  outlier_count: number;
  outliers: { line: number; image: string; value: number }[];
  /** (split mean - train mean) / train std; null without a spread. */
  shift_sd: number | null;
}

export interface TargetReportItem {
  name: string;
  range: [number, number];
  bins: number;
  /** Tukey's fences from the training split, or null without one. */
  fences: [number, number] | null;
  splits: Partial<Record<string, TargetSplit>>;
}

export interface CsvSplit {
  present: boolean;
  rows: number;
  missing_columns: string[];
  missing_images: number;
  missing_examples: { line: number; image: string }[];
  bad_examples: { line: number; target: string; cell: string }[];
}

export interface TargetReport {
  task: "regression";
  cancelled: boolean;
  tukey_k: number;
  splits: Record<string, CsvSplit>;
  targets: TargetReportItem[];
  correlations: { a: string; b: string; r: number | null }[];
}

export type TargetAnalysisJob = AnalysisJob<TargetReport>;
export type SavedTargetAnalysis = SavedAnalysis<TargetReport>;

/** A number short enough for a table cell: 4 significant digits, "—" for none. */
export function num(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  if (value !== 0 && (Math.abs(value) >= 1e5 || Math.abs(value) < 1e-3)) return value.toExponential(2);
  return String(Number(value.toPrecision(4)));
}

/** A shift of half a training standard deviation or more is worth a look. */
export const SHIFT_WARN_SD = 0.5;
