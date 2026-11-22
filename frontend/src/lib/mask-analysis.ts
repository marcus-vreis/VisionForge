/** The mask analysis of a segmentation dataset (ADR-234). */
import type { AnalysisJob, SavedAnalysis } from "./analysis-job";

export interface MaskRequest {
  base_dir: string;
  train_dir: string;
  val_dir: string;
  test_dir: string;
  images_subdir: string;
  masks_subdir: string;
  num_classes: number;
  ignore_index: number;
}

export interface MaskSplit {
  present: boolean;
  images: number;
  masks: number;
  no_mask: string[];
  no_image: string[];
  no_mask_count: number;
  no_image_count: number;
}

export interface MaskClass {
  id: number;
  ignore: boolean;
  out_of_range: boolean;
  pixels: number;
  /** Percent of every analysed pixel. */
  share: number;
  /** Percent of each split's pixels. */
  shares: Partial<Record<string, number>>;
}

export interface CoverageBin {
  lo: number;
  hi: number;
  count: number;
}

export interface MaskExample {
  split: string;
  path: string;
}

export interface MaskReport {
  task: "segmentation";
  num_classes: number;
  ignore_index: number;
  total_pairs: number;
  scanned: number;
  limit: number;
  sampled: boolean;
  cancelled: boolean;
  splits: Record<string, MaskSplit>;
  pairs: number;
  classes: MaskClass[];
  absent_classes: number[];
  /** The first bin is exactly 0 (lo = hi = 0); the others are [lo, hi) shares of the image. */
  coverage_bins: CoverageBin[];
  empty_masks: number;
  out_of_range_pixels: number;
  out_of_range_ids: number[];
  size_mismatch: number;
  color_masks: number;
  unreadable: number;
  tiny_share: number;
  tiny_regions: { id: number; regions: number }[];
  tiny_images: number;
  examples: Record<string, MaskExample[]>;
}

export type MaskAnalysisJob = AnalysisJob<MaskReport>;
export type SavedMaskAnalysis = SavedAnalysis<MaskReport>;

/** "0" for the empty bin, "1-5%" for the others, "75-100%" for the last. */
export function coverageLabel(bin: CoverageBin, empty: string): string {
  if (bin.hi === 0) return empty;
  const p = (v: number) => `${Math.round(v * 100)}`;
  return `${p(bin.lo)}–${p(bin.hi)}%`;
}

/** A present but rare class reads "<0.01%", never "0.00%", which looks absent. */
export function sharePct(value: number): string {
  if (value === 0) return "0%";
  if (value < 0.01) return "<0.01%";
  return `${value.toFixed(value < 1 ? 2 : 1)}%`;
}
