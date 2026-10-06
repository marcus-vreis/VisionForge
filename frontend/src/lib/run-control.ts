/** Stopping a running job from the interface: what the stop can promise, and
 * when the controls that reach it are on screen.
 *
 * `DELETE /api/queue/{id}` answers 200 for a running job, and ADR-088 says what
 * that means: the request was *delivered*, the trainer reads it at the top of
 * its next epoch and stops there, keeping the best checkpoint, the history so
 * far and (ADR-092/093) the state to resume from. ADR-094 is the cautionary
 * tale for a button that reports success without the other end listening, so
 * this module only lets the interface promise a stop where the backend wires
 * the cancellation token into the trainer:
 *
 * - single runs (classification, transfer learning, and the standalone tasks);
 * - classification grid / random search, which stop between trials.
 *
 * Everything else still answers 200 and keeps training — K-fold, model
 * comparison, replicates, the standalone tasks' own sweeps / K-fold /
 * comparisons, researcher-defined tasks, and PatchCore (no epochs, so no
 * boundary). Those are `"none"` here, and the interface says so instead of
 * offering a button that does nothing. When the backend wires one of them up
 * (or starts reporting a `stoppable` flag in the queue snapshot), this is the
 * one place to change.
 *
 * Pure on purpose: no DOM and no fetch, so it is covered by plain vitest.
 */

import type { QueuedJobInfo, RunStatus, TrainingEvent } from "../types/run";

/** What a stop request does to a job.
 *
 * - `epoch`: the run finishes the epoch it is in, then stops.
 * - `trial`: a multi-trial search; the trial in progress stops at its epoch
 *   boundary and the trials that have not started are skipped.
 * - `none`: the backend does not hand the token to this kind of run.
 */
export type StopMode = "epoch" | "trial" | "none";

/** The part of a queue entry that decides how it can be stopped. */
export type StopTarget = Pick<QueuedJobInfo, "task" | "strategy">;

/** Strategies that run one training and read the token at each epoch. */
const SINGLE_RUN = new Set(["simple", "classification", "transfer_learning"]);

/** Classification's multi-trial searches (the standalone tasks' sweeps arrive
 *  as `sweep:*` and are not wired). */
const TRIAL_SEARCH = new Set(["grid_search", "random_search"]);

/** How a stop request will act on `job`.
 *
 * `job` is the active entry of the queue snapshot. It is `null` while that has
 * not been read (or could not be), and then a plain single run is assumed: the
 * request itself goes through the same server, so an unreadable queue means the
 * stop would not get through either way.
 *
 * `phaseOnly` is true for a run that reports phases instead of epochs
 * (PatchCore): its memory-bank build has no epoch boundary to stop at.
 */
export function stopMode(
  job: StopTarget | null,
  opts: { phaseOnly?: boolean } = {},
): StopMode {
  if (opts.phaseOnly) return "none";
  if (job === null) return "epoch";
  if (job.task.startsWith("custom:")) return "none";
  if (SINGLE_RUN.has(job.strategy)) return "epoch";
  if (job.task === "classification" && TRIAL_SEARCH.has(job.strategy)) {
    return "trial";
  }
  return "none";
}

/** Whether the queue button is on screen.
 *
 * It used to appear only when something was waiting, so as not to keep a
 * permanent "queue 0" in front of sessions that never form one. A running job
 * is reason enough now: the row for it carries the stop control, and after a
 * page reload — which loses the training sheet — that row is the only way back
 * to a job nobody can otherwise reach. An idle server with nothing waiting
 * still shows nothing.
 */
export function showQueueButton(
  queuedCount: number,
  serverRunning: boolean,
): boolean {
  return queuedCount > 0 || serverRunning;
}

/** Whether the server is executing a job, as far as this tab can tell.
 *
 * While the tab has a run of its own, its polling is authoritative: it runs
 * while `running` and it waits behind another job while `queued`. Without one,
 * what the queue snapshot said on load (and keeps saying while it polls) is the
 * best answer there is.
 */
export function serverRunning(args: {
  runActive: boolean;
  status: RunStatus["status"];
  seededRunning: boolean;
}): boolean {
  if (args.runActive) {
    return args.status === "running" || args.status === "queued";
  }
  return args.seededRunning;
}

/** Whether a run ended before its last epoch (or, for a search, before its last
 * trial), judged from the events it streamed.
 *
 * The backend reports a stopped run as an ordinary `completed` one — the
 * cancellation is not part of the status — so the only evidence is that the
 * count stopped short of what was configured. A run that early-stops on its own
 * looks the same; this is only read after the researcher asked for a stop.
 */
export function endedEarly(events: readonly TrainingEvent[]): boolean {
  let lastEpoch: Extract<TrainingEvent, { event: "epoch_end" }> | undefined;
  let plannedTrials = 0;
  let ranTrials: number | undefined;
  for (const e of events) {
    if (e.event === "epoch_end") lastEpoch = e;
    else if (e.event === "trial_start") plannedTrials = e.total_trials;
    else if (e.event === "end") ranTrials = e.total_trials;
  }
  if (ranTrials !== undefined && plannedTrials > 0 && ranTrials < plannedTrials) {
    return true;
  }
  return lastEpoch !== undefined && lastEpoch.epoch < lastEpoch.total_epochs;
}

/** Where the stream's last epoch stood, for the "stopped at" line. */
export function lastEpochOf(
  events: readonly TrainingEvent[],
): { epoch: number; total: number } | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    if (e.event === "epoch_end") return { epoch: e.epoch, total: e.total_epochs };
  }
  return null;
}

/** Whether the stream carried its terminal `end` event. */
export function hasEnded(events: readonly TrainingEvent[]): boolean {
  return events.some((e) => e.event === "end");
}

/** What became of a stop request.
 *
 * - `none`: nothing was asked.
 * - `stopping`: delivered, and the run is still going — it finishes its epoch.
 * - `stopped`: the run ended short of its configured length.
 * - `too-late`: the run ended, but it had already reached its last epoch, so
 *   the request changed nothing and it completed normally.
 */
export type StopOutcome = "none" | "stopping" | "stopped" | "too-late";

export function stopOutcome(args: {
  requested: boolean;
  ended: boolean;
  events: readonly TrainingEvent[];
}): StopOutcome {
  if (!args.requested) return "none";
  if (!args.ended) return "stopping";
  return endedEarly(args.events) ? "stopped" : "too-late";
}
