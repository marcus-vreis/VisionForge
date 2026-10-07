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
 * (or starts reporting what a job can do in the queue snapshot), `stopMode` is
 * the one place to change: it already takes the snapshot's job first and falls
 * back to what the training sheet itself knows only when there is none.
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
 * - `unconfirmed`: an anomaly run that has not yet shown whether it trains by
 *   epochs (an autoencoder) or builds a memory bank (PatchCore, which has no
 *   boundary to stop at). Not offered until the stream says which.
 */
export type StopMode = "epoch" | "trial" | "none" | "unconfirmed";

/** The part of a queue entry that decides how it can be stopped. */
export type StopTarget = Pick<QueuedJobInfo, "task" | "strategy">;

/** Whether a run is made of epochs, as far as its stream has shown. */
export type EpochLoop = "yes" | "no" | "unknown";

/** What the training sheet knows about its own run without asking the server:
 * the block recorded when the run was submitted, the tab it was submitted from,
 * and what its stream has shown so far. */
export interface LocalRun {
  /** App's `blockKind`: `classification`, `grid_search`, `detection`, `custom`… */
  blockKind: string;
  /** App's active tab. Only used to tell a classification search from another
   *  task's sweep, which share the block name. */
  taskKey: string;
  epochLoop: EpochLoop;
}

/** Strategies that run one training and read the token at each epoch. */
const SINGLE_RUN = new Set(["simple", "classification", "transfer_learning"]);

/** Classification's multi-trial searches (the standalone tasks' sweeps arrive
 *  as `sweep:*` and are not wired). */
const TRIAL_SEARCH = new Set(["grid_search", "random_search"]);

/** Blocks the classification tab submits under its own name. */
const CLASSIFICATION_BLOCKS = new Set([
  "classification",
  "transfer_learning",
  "cross_validation",
  "model_comparison",
]);

/** Blocks the standalone tabs submit for a plain run. */
const STANDALONE_TASKS = new Set([
  "detection",
  "regression",
  "segmentation",
  "anomaly",
]);

/** The queue entry a run would have, rebuilt from what the sheet recorded. */
function targetFromLocal(local: LocalRun): StopTarget {
  const { blockKind, taskKey } = local;
  if (blockKind === "custom") return { task: `custom:${taskKey}`, strategy: "simple" };
  if (STANDALONE_TASKS.has(blockKind)) return { task: blockKind, strategy: "simple" };
  if (CLASSIFICATION_BLOCKS.has(blockKind)) {
    return { task: "classification", strategy: blockKind };
  }
  if (TRIAL_SEARCH.has(blockKind)) {
    // The two paths share the block name, and only the tab tells them apart.
    return taskKey === "classification"
      ? { task: "classification", strategy: blockKind }
      : { task: taskKey, strategy: "sweep" };
  }
  // Replicates, and anything the sheet has not been taught: not promised.
  return { task: taskKey, strategy: blockKind };
}

/** How a stop request will act on a run.
 *
 * `job` is the active entry of the queue snapshot, which is the authority. It
 * is `null` while that has not been read, or could not be; `local` is then what
 * the training sheet knows of its own run, so an unreadable queue does not turn
 * a K-fold into a "plain run" with a stop button the server ignores. With
 * neither, a plain run is assumed.
 *
 * `local.epochLoop` is what the run's own stream has shown. A run that reports
 * phases instead of epochs (PatchCore) has no boundary to stop at, and an
 * anomaly run is not offered the stop until its stream has said which kind it
 * is — the queue entry only says "anomaly", which covers both.
 */
export function stopMode(job: StopTarget | null, local?: LocalRun): StopMode {
  if (local?.epochLoop === "no") return "none";
  const target = job ?? (local ? targetFromLocal(local) : null);
  if (target === null) return "epoch";
  if (target.task === "anomaly" && local && local.epochLoop !== "yes") {
    return "unconfirmed";
  }
  if (target.task.startsWith("custom:")) return "none";
  if (SINGLE_RUN.has(target.strategy)) return "epoch";
  if (target.task === "classification" && TRIAL_SEARCH.has(target.strategy)) {
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

type EpochEnd = Extract<TrainingEvent, { event: "epoch_end" }>;
type StartEvent = Extract<TrainingEvent, { event: "start" }>;
type EndEvent = Extract<TrainingEvent, { event: "end" }>;

/** The events a verdict on how a run ended is drawn from. */
function milestones(events: readonly TrainingEvent[]): {
  start: StartEvent | undefined;
  lastEpoch: EpochEnd | undefined;
  end: EndEvent | undefined;
  plannedTrials: number;
} {
  let start: StartEvent | undefined;
  let lastEpoch: EpochEnd | undefined;
  let end: EndEvent | undefined;
  let plannedTrials = 0;
  for (const e of events) {
    if (e.event === "start") start = e;
    else if (e.event === "epoch_end") lastEpoch = e;
    else if (e.event === "trial_start") plannedTrials = e.total_trials;
    else if (e.event === "end") end = e;
  }
  return { start, lastEpoch, end, plannedTrials };
}

/** How a run ended, judged from the events it streamed: short of what was
 * configured (`early`), at its full length (`complete`), or without a stream
 * that says (`unknown`).
 *
 * The backend reports a stopped run as an ordinary `completed` one — the
 * cancellation is not part of the status — so the only evidence is the count
 * stopping short. `complete` needs an epoch report that reached its total: an
 * empty or partial stream is not proof that a run finished, and a stopped run
 * must never be told it finished normally.
 *
 * A resumed run does not replay the epochs of its earlier pass, so one stopped
 * before it trained another epoch streams only `start` (how long it was meant
 * to be) and `end` (how far its history got); that comparison is the evidence
 * when no epoch was reported. A run that early-stops on its own looks the same
 * as a stopped one; this is only read after the researcher asked for a stop.
 */
export function runEnding(
  events: readonly TrainingEvent[],
): "early" | "complete" | "unknown" {
  const { start, lastEpoch, end, plannedTrials } = milestones(events);
  if (plannedTrials > 0) {
    // A search's `end` counts the trials that ran, and its epochs are 0.
    if (end?.total_trials !== undefined && end.total_trials < plannedTrials) {
      return "early";
    }
  } else if (lastEpoch === undefined && start !== undefined && end !== undefined) {
    return end.total_epochs < start.total_epochs ? "early" : "unknown";
  }
  if (lastEpoch === undefined) return "unknown";
  return lastEpoch.epoch < lastEpoch.total_epochs ? "early" : "complete";
}

/** Where the run stood when it ended, for the "stopped at" line: its last
 * reported epoch, or — for a run that reported none — how far its history got
 * out of how long it was meant to be. `null` when the stream does not say. */
export function reachedEpoch(
  events: readonly TrainingEvent[],
): { epoch: number; total: number } | null {
  const { start, lastEpoch, end, plannedTrials } = milestones(events);
  if (lastEpoch !== undefined) {
    return { epoch: lastEpoch.epoch, total: lastEpoch.total_epochs };
  }
  if (plannedTrials === 0 && start !== undefined && end !== undefined) {
    return { epoch: end.total_epochs, total: start.total_epochs };
  }
  return null;
}

/** Whether the stream has shown that the run is made of epochs.
 *
 * PatchCore reports phases, then closes with a single `epoch_end`, so a phase
 * settles it as `no` whatever follows. Several configured epochs or an epoch
 * report settle it as `yes`; a lone configured epoch cannot tell an
 * autoencoder from PatchCore's one step, so it stays `unknown`.
 */
export function epochLoop(events: readonly TrainingEvent[]): EpochLoop {
  let epochs: number | null = null;
  let sawEpoch = false;
  for (const e of events) {
    if (e.event === "phase") return "no";
    if (e.event === "start") epochs = e.total_epochs;
    if (e.event === "epoch_end") sawEpoch = true;
  }
  if (sawEpoch || (epochs !== null && epochs > 1)) return "yes";
  return "unknown";
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
 * - `too-late`: the run ended, but its last epoch report reached the total, so
 *   the request changed nothing and it completed normally.
 * - `unknown`: the run ended and the stream does not say how far it got, so
 *   nothing is claimed either way.
 */
export type StopOutcome = "none" | "stopping" | "stopped" | "too-late" | "unknown";

export function stopOutcome(args: {
  requested: boolean;
  ended: boolean;
  events: readonly TrainingEvent[];
}): StopOutcome {
  if (!args.requested) return "none";
  if (!args.ended) return "stopping";
  switch (runEnding(args.events)) {
    case "early":
      return "stopped";
    case "complete":
      return "too-late";
    default:
      return "unknown";
  }
}
