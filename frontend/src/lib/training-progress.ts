import type { TrainingEvent } from "../types/run";

/** How far into its unit (one training, or the trial in flight) the run is,
 *  read from the latest `batch` event, or `null` when there is none to use.
 *
 *  A `batch` only counts while it is the newest word about its unit: an
 *  `epoch_end`, `trial_start` or `trial_end` after it says more. Without this
 *  the overlay moved only at `epoch_end` and sat still for a whole long epoch.
 *
 *  An epoch's last batch is always reported (ADR-205) and its validation runs
 *  after it, so that batch is held half a batch short of the epoch's end: the
 *  bar would otherwise say 100% while the last epoch still validates. */
export function liveBatchFraction(events: readonly TrainingEvent[]): number | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (e.event === "batch") {
      if (e.total_epochs <= 0 || e.total_batches <= 0) return null;
      const batchesDone = Math.min(e.batch, e.total_batches - 0.5);
      const within = (e.epoch - 1 + batchesDone / e.total_batches) / e.total_epochs;
      return Math.min(1, Math.max(0, within));
    }
    if (
      e.event === "epoch_end" ||
      e.event === "trial_start" ||
      e.event === "trial_end" ||
      e.event === "end"
    ) {
      return null;
    }
  }
  return null;
}
