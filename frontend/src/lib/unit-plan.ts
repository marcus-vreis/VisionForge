/** How many units a multi-unit job planned, against how many of them ran.
 *
 * A stopped K-fold, comparison, search or replicate set lists only the units
 * that ran, so a header that counts "ok/total" over that list says "0/1" for a
 * 5-fold job stopped in its first fold, while the training sheet says "0/5". The
 * header counts against what was planned, and says how many never started.
 */

/** The units a report was asked to run, when the report or the submission says.
 *
 * The report says it in four ways: a K-fold of the other tasks names its folds
 * (`n_folds`), a standalone sweep its trials (`planned_trials`), and a replicate
 * set the seeds it was asked for (`seeds`). The classification K-fold, the
 * comparisons and the grid search list only what ran, so for them the count the
 * page recorded at submission stands in. `null` when nothing says. */
export function plannedUnits(
  report: Record<string, unknown>,
  submitted: number | null | undefined,
): number | null {
  const folds = report["n_folds"];
  if (typeof folds === "number") return folds;
  const trials = report["planned_trials"];
  if (typeof trials === "number") return trials;
  const seeds = report["seeds"];
  if (Array.isArray(seeds)) return seeds.length;
  return typeof submitted === "number" ? submitted : null;
}

/** The denominator of a header, and how many units never started.
 *
 * A plan smaller than what ran is a stale or wrong figure, not a result: the
 * count falls back to the units that ran, so a header never reads "3/2". */
export function unitPlan(
  ran: number,
  planned: number | null | undefined,
): { total: number; notRun: number } {
  if (typeof planned === "number" && Number.isFinite(planned) && planned >= ran && planned > 0) {
    return { total: planned, notRun: planned - ran };
  }
  return { total: ran, notRun: 0 };
}
