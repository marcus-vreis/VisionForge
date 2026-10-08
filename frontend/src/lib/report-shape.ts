/** Which report a result carries, told by the shape of its fields.
 *
 * The result view picks its layout from the report, so the test for a layout has
 * to hold for the reports a stopped job writes too: a K-fold stopped after one
 * finished fold has no spread, and one stopped before any finished has no mean.
 */

/** The classification K-fold report (`CrossValidationBlock.report()`).
 *
 * It lists its folds and carries `mean_accuracy` and the other aggregates — as
 * numbers when enough folds finished, `null` when too few did (no mean before
 * one fold, no spread before two). The standalone tasks' K-fold report also lists
 * `fold_results`, but its figures live under `aggregate` and it has no
 * `mean_accuracy`, which is what tells them apart. */
export function isCrossValidationReport(report: Record<string, unknown>): boolean {
  return Array.isArray(report["fold_results"]) && "mean_accuracy" in report;
}
