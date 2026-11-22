/** Which classes look alike before training (ADR-239). */
import type { AnalysisJob, SavedAnalysis } from "./analysis-job";
import type { ExplorerForm } from "./dataset-explorer";

export interface ClassSimilarityRequest extends ExplorerForm {
  per_class: number;
}

export interface SimilarityExample {
  path: string;
  class: string;
  neighbour: string;
  neighbour_class: string;
  similarity: number;
}

export interface SimilarityPair {
  a: string;
  b: string;
  /** Share of a's images whose nearest image is a b. */
  a_to_b: number;
  b_to_a: number;
  centroid_similarity: number;
  flagged: boolean;
  examples: SimilarityExample[];
}

export interface ClassSimilarityReport {
  model: string;
  per_class: number;
  images_per_class: Record<string, number>;
  sampled: boolean;
  cancelled: boolean;
  unreadable: number;
  described: number;
  classes: { name: string; used: number; purity: number | null }[];
  flag_share: number;
  flagged_count: number;
  pair_count: number;
  pairs: SimilarityPair[];
}

export type ClassSimilarityJob = AnalysisJob<ClassSimilarityReport>;
export type SavedClassSimilarity = SavedAnalysis<ClassSimilarityReport>;

/** 0.3333 -> "33%". */
export function sharePct(share: number): string {
  return `${Math.round(share * 100)}%`;
}
