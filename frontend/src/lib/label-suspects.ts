/** Suspicious labels of a classification run (ADR-237). */

export type SuspectSplit = "train" | "val" | "test";

export interface Suspect {
  /** Relative to the split folder: `<class>/<file>`. */
  path: string;
  given: string;
  suspected: string;
  p_given: number;
  p_suspected: number;
}

export interface SuspectSplitReport {
  folder: string;
  images: number;
  failed: number;
  /** Training images: the model saw them, so it agrees with what it memorized. */
  in_sample: boolean;
  thresholds: (number | null)[];
  suspect_count: number;
  suspects: Suspect[];
  joint: number[][];
  pairs: { given: string; taken_for: string; images: number }[];
}

export interface LabelSuspectsReport {
  run_dir: string;
  computed_at: string;
  class_names: string[];
  splits: Partial<Record<SuspectSplit, SuspectSplitReport>>;
  missing_splits: SuspectSplit[];
}
