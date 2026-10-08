/** The line a stopped run writes at the end of its log.
 *
 * What it says depends on how the job stopped (`StopSummary`): the epoch a
 * single run reached, how many folds / models / replicates / trials of a
 * multi-unit job finished, or what a PatchCore kept. Taking the dictionary, it
 * follows the language of the reader.
 */

import type { Dict } from "../i18n/pt";
import type { StopSummary } from "./run-control";

export function stopLogLine(t: Dict, summary: StopSummary): string {
  const words = t.trainingOverlay;
  switch (summary.kind) {
    case "units":
      return words.stoppedUnitsLog(
        summary.unit,
        summary.finished,
        summary.planned,
        summary.stopped,
      );
    case "phase":
      return words.stoppedPhaseLog(summary.bankKept);
    default:
      return words.stoppedLog(summary.epoch, summary.total);
  }
}
