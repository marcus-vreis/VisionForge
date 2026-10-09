/**
 * What the History comparison owes the reader about seeds (ADR-112, ADR-113).
 *
 * A run in History is one training under one seed, and seed-to-seed variance
 * routinely exceeds the gap between two configurations. A table that puts two
 * such runs side by side and marks the larger number is answering "which is
 * better?" with one sample of each; the only honest answer from there is "this
 * one is larger, and that may be the seed".
 *
 * A replicate group is the other kind of run (ADR-113): its cell is a mean over
 * seeds with the interval the report gave, which carries its own uncertainty, so
 * the single-seed caution does not apply to it. The note is about the runs that
 * are one seed. It is judged on those alone: two groups side by side raise no
 * note, and a group beside single runs raises it only for a gap among the
 * single runs, in words that say which runs it speaks of. The frontend still
 * computes no statistics of its own.
 */
import { extremeIndexes, type MetricDirection } from "./compare-metrics";

/** The caution the table carries: every run is a single seed, or only some are. */
export type SeedNote = "single-seed" | "single-seed-mixed";

export interface SeedRow {
  direction: MetricDirection | null;
  /** One entry per compared run; null where the run did not measure it. */
  values: ReadonlyArray<number | null>;
}

/** A row's values with the group runs taken out (set to null), so the note is
 *  judged on the single-seed runs only. `isGroup` is parallel to `values`. */
export function singleSeedValues(
  values: ReadonlyArray<number | null>,
  isGroup: ReadonlyArray<boolean>,
): Array<number | null> {
  return values.map((value, i) => (isGroup[i] ? null : value));
}

/**
 * Which note goes under the metric table, if any.
 *
 * The note is about a gap, so it appears exactly when the table shows one: some
 * row marks a highest or lowest value (two or more runs measured it, it has a
 * direction, and the values differ). A table with no such row implies no winner
 * and warns of none, rather than repeating itself under every comparison.
 * `rows` carry the single-seed runs' values (`singleSeedValues`); `hasGroups`
 * only picks the wording, because "each run has a single seed" is false when
 * some of the compared runs are means over seeds.
 */
export function seedNote(rows: ReadonlyArray<SeedRow>, hasGroups = false): SeedNote | null {
  const gap = rows.some((row) => extremeIndexes(row.values, row.direction).length > 0);
  if (!gap) return null;
  return hasGroups ? "single-seed-mixed" : "single-seed";
}
