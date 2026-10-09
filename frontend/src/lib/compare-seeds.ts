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
 * the single-seed caution does not apply to the group's own cell. The note is
 * about the runs that are one seed, and it is judged on the cells the table
 * actually highlights: a row warns when it marks a highest or lowest value
 * (group means included, since they are on the same row) and at least one run
 * on that row is a single seed. Two groups side by side raise no note; one
 * single run beside a group mean does, because the highlighted gap then rests
 * on one seed. The wording says which runs it speaks of. The frontend still
 * computes no statistics of its own.
 */
import { extremeIndexes, type MetricDirection } from "./compare-metrics";

/** The caution the table carries: every run is a single seed, or only some are. */
export type SeedNote = "single-seed" | "single-seed-mixed";

export interface SeedRow {
  direction: MetricDirection | null;
  /** One entry per compared run, replicate-group means included; null where the
   *  run did not measure it. These are the values the table marks an extreme
   *  among, so the note is judged on the same ones. */
  values: ReadonlyArray<number | null>;
}

/**
 * Which note goes under the metric table, if any.
 *
 * The note is about a gap, so it appears exactly when the table shows one: some
 * row marks a highest or lowest value (two or more runs measured it, it has a
 * direction, and the values differ) and a single-seed run measured that row. A
 * table with no such row implies no winner and warns of none, rather than
 * repeating itself under every comparison.
 *
 * `isGroup` is parallel to each row's `values` (true where the run is a
 * replicate group, a mean over seeds); omitted, every run is one seed. It picks
 * the wording too, because "each run has a single seed" is false when some of
 * the compared runs are means over seeds.
 */
export function seedNote(
  rows: ReadonlyArray<SeedRow>,
  isGroup: ReadonlyArray<boolean> = [],
): SeedNote | null {
  const warns = rows.some(
    (row) =>
      extremeIndexes(row.values, row.direction).length > 0 &&
      row.values.some((value, i) => value !== null && !isGroup[i]),
  );
  if (!warns) return null;
  return isGroup.some(Boolean) ? "single-seed-mixed" : "single-seed";
}
