/**
 * What the History comparison owes the reader about seeds (ADR-112).
 *
 * A run in History is one training under one seed, and seed-to-seed variance
 * routinely exceeds the gap between two configurations. A table that puts two
 * such runs side by side and marks the larger number is answering "which is
 * better?" with one sample of each; the only honest answer from there is "this
 * one is larger, and that may be the seed".
 *
 * Replicates and replicated comparisons are where the real answer lives (mean
 * ± CI, paired tests with Holm), but they keep it in a report on disk, not as a
 * run in History, so nothing the panel compares can be a group of seeds. When
 * that changes, the groups join the decision here; the frontend still computes
 * no statistics of its own.
 */
import { extremeIndexes, type MetricDirection } from "./compare-metrics";

/** The caution the table carries. One kind today: every run is a single seed. */
export type SeedNote = "single-seed";

export interface SeedRow {
  direction: MetricDirection | null;
  /** One entry per compared run; null where the run did not measure it. */
  values: ReadonlyArray<number | null>;
}

/**
 * Which note goes under the metric table, if any.
 *
 * The note is about a gap, so it appears exactly when the table shows one: some
 * row marks a highest or lowest value (two or more runs measured it, it has a
 * direction, and the values differ). A table with no such row implies no winner
 * and warns of none, rather than repeating itself under every comparison.
 */
export function seedNote(rows: ReadonlyArray<SeedRow>): SeedNote | null {
  const gap = rows.some((row) => extremeIndexes(row.values, row.direction).length > 0);
  return gap ? "single-seed" : null;
}
