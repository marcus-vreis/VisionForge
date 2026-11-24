/** The analysis of an anomaly-detection dataset (ADR-236). */
import type { AnalysisJob, SavedAnalysis } from "./analysis-job";
import type { QualityReport } from "./dataset-explorer";

export interface AnomalyRequest {
  base_dir: string;
  train_dir: string;
  test_dir: string;
  normal_dir: string;
}

export interface AnomalyGroup {
  /** "train", "test_normal" or "defect:<kind>". */
  group: string;
  sizes: { size: string; count: number }[];
  distinct_sizes: number;
  modes: Record<string, number>;
}

export interface LookAlike {
  path: string;
  like: string;
  distance: number;
}

export interface AnomalyReport {
  task: "anomaly";
  total_images: number;
  scanned: number;
  limit: number;
  sampled: boolean;
  cancelled: boolean;
  max_distance: number;
  few_images: number;
  train_present: boolean;
  test_present: boolean;
  defect_kinds: string[];
  extra_train_folders: string[];
  ground_truth: boolean;
  counts: Record<string, number>;
  groups: AnomalyGroup[];
  size_mismatch: string[];
  mode_mismatch: string[];
  train_like_defect_count: number;
  train_like_defect: LookAlike[];
  test_like_train_count: number;
  test_like_train: LookAlike[];
  missing_masks: { count: number; paths: string[] } | null;
  unreadable_count: number;
  unreadable: { group: string; path: string }[];
  /** Brightness, contrast and sharpness of the images (ADR-240); absent from older results. */
  quality?: QualityReport;
}

export type AnomalyAnalysisJob = AnalysisJob<AnomalyReport>;
export type SavedAnomalyAnalysis = SavedAnalysis<AnomalyReport>;

export const DEFECT_PREFIX = "defect:";
