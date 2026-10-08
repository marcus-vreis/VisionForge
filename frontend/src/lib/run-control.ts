/** Stopping a running job from the interface: where a stop lands, what it leaves
 * behind, and when the controls that reach it are on screen.
 *
 * `DELETE /api/queue/{id}` answers 200 for a running job, and ADR-088 says what
 * that means: the request was *delivered*, and the job stops at the next
 * boundary it reads the stop at, keeping what it has earned. Since ADR-111 every
 * kind of job reads it, and the queue says where: `stop_at` on each entry —
 * the epoch of a single run, the trial of a search, the fold of a K-fold, the
 * model of a comparison, the replicate of a replicate set, or the phase of
 * PatchCore. A job that owns its own loop (a custom task, Level 2) has no such
 * boundary: its `stop_at` is `null` and DELETE refuses it with 409.
 *
 * So `stopMode` is the queue's word whenever the queue gives it. The rest of
 * this module exists for when it does not: the mapping from a strategy to its
 * stop point (mirroring `_STOP_POINTS` in `gui/api/routes.py`) for a snapshot
 * that omits `stop_at`, and the training sheet's own record of its run for a
 * snapshot that could not be read.
 *
 * Pure on purpose: no DOM and no fetch, so it is covered by plain vitest.
 */

import type {
  QueuedJobInfo,
  RunStatus,
  StopPoint,
  TrainingEvent,
} from "../types/run";

export type { StopPoint };

/** How a stop request acts on a job: a stop point, or why there is none.
 *
 * - a stop point: the job finishes what is in flight to that boundary and
 *   starts nothing new;
 * - `none`: the job cannot be stopped once it runs (`stop_at` is `null`);
 * - `unconfirmed`: not known yet — a custom task whose level only the queue
 *   knows, or an anomaly run that has not shown whether it trains by epochs
 *   (an autoencoder) or builds a memory bank (PatchCore). Not offered until
 *   it is.
 */
export type StopMode = StopPoint | "none" | "unconfirmed";

/** The part of a queue entry that decides how it can be stopped. */
export type StopTarget = Pick<QueuedJobInfo, "task" | "strategy" | "stop_at">;

/** The stop points that mean "a unit of a multi-unit job". */
export type UnitKind = "trial" | "fold" | "model" | "replicate";

/** Whether a run is made of epochs, as far as its stream has shown. */
export type EpochLoop = "yes" | "no" | "unknown";

/** What the training sheet knows about its own run without asking the server:
 * the block recorded when the run was submitted, and what its stream has shown
 * so far. */
export interface LocalRun {
  /** App's `blockKind`: `classification`, `grid_search`, `detection`, `custom`… */
  blockKind: string;
  epochLoop: EpochLoop;
}

/** The server's table (`_STOP_POINTS` in gui/api/routes.py), by the strategy a
 *  job is queued under. A strategy missing from it is not stoppable there. */
const STOP_POINTS: Record<string, StopPoint> = {
  simple: "epoch",
  classification: "epoch",
  transfer_learning: "epoch",
  batch_prediction: "epoch",
  export_onnx: "epoch",
  grid_search: "trial",
  random_search: "trial",
  sweep: "trial",
  cross_validation: "fold",
  cv: "fold",
  model_comparison: "model",
  comparison: "model",
  replicates: "replicate",
  "replicated-comparison": "replicate",
};

/** Blocks the standalone tabs submit for a plain run. */
const STANDALONE_TASKS = new Set([
  "detection",
  "regression",
  "segmentation",
  "anomaly",
]);

/** The queue entry a run would have, rebuilt from what the sheet recorded. */
function targetFromLocal(local: LocalRun): Pick<StopTarget, "task" | "strategy"> {
  const { blockKind } = local;
  if (blockKind === "custom") return { task: "custom:", strategy: "simple" };
  if (STANDALONE_TASKS.has(blockKind)) return { task: blockKind, strategy: "simple" };
  // Classification files its strategy under the block name, and the standalone
  // tasks' sweeps, K-fold and replicates are named alike: the table does not
  // care which tab they came from.
  return { task: "", strategy: blockKind };
}

/** How a stop request acts on a run.
 *
 * `job` is the active entry of the queue snapshot, which is the authority: its
 * `stop_at` is the answer, `null` meaning the job cannot be stopped. When the
 * snapshot is missing, or does not carry `stop_at`, the answer is rebuilt from
 * the strategy (and, without a snapshot, from `local`, the block the sheet
 * recorded at submission) with the server's own table. With neither, a plain run
 * is assumed.
 *
 * Two kinds of run cannot be told from the strategy: a custom task, whose level
 * only the snapshot knows, and an anomaly run, which may be PatchCore. Both are
 * `unconfirmed` until the snapshot says; for anomaly, `local.epochLoop` can say
 * it sooner, from the stream.
 */
export function stopMode(job: StopTarget | null, local?: LocalRun): StopMode {
  if (job && job.stop_at !== undefined) return job.stop_at ?? "none";
  const target = job ?? (local ? targetFromLocal(local) : null);
  if (target === null) return "epoch";
  if (target.strategy === "simple") {
    if (target.task.startsWith("custom:")) return "unconfirmed";
    if (target.task === "anomaly" && local) {
      if (local.epochLoop === "yes") return "epoch";
      return local.epochLoop === "no" ? "phase" : "unconfirmed";
    }
  }
  return STOP_POINTS[target.strategy.split(":", 1)[0]] ?? "none";
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

/** The events a verdict on how a run ended is drawn from.
 *
 * A search streams many trainings in one stream, so the last epoch report of
 * the stream is only the last epoch of the *last trial that trained* — which is
 * not the last trial when that one was stopped before its first epoch. Epochs
 * are therefore also kept per trial, by the index the backend stamps on them or,
 * when it is missing, by the trial that was started most recently.
 */
function milestones(events: readonly TrainingEvent[]): {
  start: StartEvent | undefined;
  lastEpoch: EpochEnd | undefined;
  end: EndEvent | undefined;
  plannedTrials: number;
  /** Index of the trial started most recently; -1 outside a search. */
  currentTrial: number;
  /** The last epoch report of each trial that reported one. */
  epochByTrial: Map<number, EpochEnd>;
  /** How many epochs each trial ran, from its `trial_end`. */
  trialEpochs: Map<number, number>;
  sawPhase: boolean;
} {
  let start: StartEvent | undefined;
  let lastEpoch: EpochEnd | undefined;
  let end: EndEvent | undefined;
  let plannedTrials = 0;
  let currentTrial = -1;
  let sawPhase = false;
  const epochByTrial = new Map<number, EpochEnd>();
  const trialEpochs = new Map<number, number>();
  for (const e of events) {
    if (e.event === "start") start = e;
    else if (e.event === "epoch_end") {
      lastEpoch = e;
      epochByTrial.set(e.trial_index ?? currentTrial, e);
    } else if (e.event === "trial_start") {
      plannedTrials = e.total_trials;
      currentTrial = e.trial_index;
    } else if (e.event === "trial_end") trialEpochs.set(e.trial_index, e.total_epochs);
    else if (e.event === "phase") sawPhase = true;
    else if (e.event === "end") end = e;
  }
  return {
    start,
    lastEpoch,
    end,
    plannedTrials,
    currentTrial,
    epochByTrial,
    trialEpochs,
    sawPhase,
  };
}

/** How a run ended, judged from the events it streamed: short of what was
 * configured (`early`), at its full length (`complete`), or without a stream
 * that says (`unknown`).
 *
 * This is the inference for when nothing explicit exists. The backend reports a
 * stopped run as an ordinary `completed` one — the cancellation is not part of
 * the status — so the evidence is the count stopping short. `complete` needs an
 * epoch report that reached its total: an empty or partial stream is not proof
 * that a run finished, and a stopped run must never be told it finished
 * normally. A multi-unit job names its stopped unit in its report, which
 * `stopOutcome` reads first (`unitCounts`).
 *
 * A resumed run does not replay the epochs of its earlier pass, so one stopped
 * before it trained another epoch streams only `start` (how long it was meant
 * to be) and `end` (how far its history got); that comparison is the evidence
 * when no epoch was reported. PatchCore reports an epoch only once it has
 * scored, so phases followed by an `end` with no epoch are a bank that was never
 * scored. A run that early-stops on its own looks the same as a stopped one;
 * this is only read after the researcher asked for a stop.
 *
 * A search is `early` when it skipped trials, or when its *last planned trial*
 * ran no epoch or stopped short of its total, and `complete` when that trial
 * reached its total. Earlier trials are not asked: one that stopped on patience
 * is a finished trial.
 */
export function runEnding(
  events: readonly TrainingEvent[],
): "early" | "complete" | "unknown" {
  const { start, lastEpoch, end, plannedTrials, epochByTrial, trialEpochs, sawPhase } =
    milestones(events);
  if (plannedTrials > 0) {
    // A search's `end` counts the trials that ran, and its epochs are 0.
    if (end?.total_trials !== undefined && end.total_trials < plannedTrials) {
      return "early";
    }
    const last = plannedTrials - 1;
    if (trialEpochs.get(last) === 0) return "early";
    const finalEpoch = epochByTrial.get(last);
    if (finalEpoch === undefined) return "unknown";
    return finalEpoch.epoch < finalEpoch.total_epochs ? "early" : "complete";
  }
  if (lastEpoch === undefined) {
    if (start !== undefined && end !== undefined && end.total_epochs < start.total_epochs) {
      return "early";
    }
    return sawPhase && end !== undefined ? "early" : "unknown";
  }
  return lastEpoch.epoch < lastEpoch.total_epochs ? "early" : "complete";
}

/** Where the run stood when it ended, for the "stopped at" line: its last
 * reported epoch, or — for a run that reported none — how far its history got
 * out of how long it was meant to be. For a search it is the trial that was
 * started last: an earlier trial's final epoch says nothing about where a trial
 * stopped before its first epoch got to. `null` when the stream does not say. */
export function reachedEpoch(
  events: readonly TrainingEvent[],
): { epoch: number; total: number } | null {
  const { start, lastEpoch, end, plannedTrials, currentTrial, epochByTrial, trialEpochs } =
    milestones(events);
  if (plannedTrials > 0) {
    const latest = epochByTrial.get(currentTrial);
    if (latest !== undefined) return { epoch: latest.epoch, total: latest.total_epochs };
    if (trialEpochs.get(currentTrial) === 0) {
      return { epoch: 0, total: lastEpoch?.total_epochs ?? 0 };
    }
    return null;
  }
  if (lastEpoch !== undefined) {
    return { epoch: lastEpoch.epoch, total: lastEpoch.total_epochs };
  }
  if (start !== undefined && end !== undefined) {
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

/** How many units of a multi-unit job finished, were cut by a stop, or failed. */
export interface UnitCounts {
  finished: number;
  stopped: number;
  failed: number;
  total: number;
}

/** The units a result report lists, counted by the status the server gave them.
 *
 * After ADR-111 a unit that was running when the stop arrived is recorded with
 * `status: "stopped"` — in `fold_results` (K-fold) and in `trials` (sweeps,
 * comparisons, replicates) — and the classification comparison, which lists
 * only its top three, reports `stopped_count` instead. This is the explicit
 * evidence that a job was stopped; `null` is a report that does not list its
 * units (a single run, or the classification grid search, which reports only
 * totals).
 */
export function unitCounts(
  report: Record<string, unknown> | null | undefined,
): UnitCounts | null {
  if (!report) return null;
  const rows = Array.isArray(report["fold_results"])
    ? (report["fold_results"] as unknown[])
    : Array.isArray(report["trials"])
      ? (report["trials"] as unknown[])
      : null;
  if (rows !== null) {
    const status = (row: unknown): unknown =>
      typeof row === "object" && row !== null
        ? (row as Record<string, unknown>)["status"]
        : undefined;
    const finished = rows.filter((r) => status(r) === "success").length;
    const stopped = rows.filter((r) => status(r) === "stopped").length;
    return { finished, stopped, failed: rows.length - finished - stopped, total: rows.length };
  }
  const total = report["total_ran"];
  const stopped = report["stopped_count"];
  if (typeof total === "number" && typeof stopped === "number") {
    const failedCount = report["failed_count"];
    const failed = typeof failedCount === "number" ? failedCount : 0;
    return { finished: Math.max(0, total - stopped - failed), stopped, failed, total };
  }
  return null;
}

/** What became of a stop request.
 *
 * - `none`: nothing was asked.
 * - `stopping`: delivered, and the run is still going — it finishes its epoch.
 * - `stopped`: the run ended short of its configured length, or the report
 *   names a unit the server cut.
 * - `too-late`: the run ended, but its last epoch report reached the total, so
 *   the request changed nothing and it completed normally.
 * - `unknown`: the run ended and nothing says how far it got, so nothing is
 *   claimed either way.
 */
export type StopOutcome = "none" | "stopping" | "stopped" | "too-late" | "unknown";

/** The outcome of a stop request. The explicit evidence comes first: a report
 * that names a stopped unit settles it, even when the stream looks complete (a
 * unit cut during its last epoch is still recorded as stopped). Only without it
 * is the stream read (`runEnding`). `report` is the result payload, which is not
 * there yet while the run is closing. */
export function stopOutcome(args: {
  requested: boolean;
  ended: boolean;
  events: readonly TrainingEvent[];
  report?: Record<string, unknown> | null;
}): StopOutcome {
  if (!args.requested) return "none";
  if (!args.ended) return "stopping";
  const counts = unitCounts(args.report);
  if (counts !== null && counts.stopped > 0) return "stopped";
  switch (runEnding(args.events)) {
    case "early":
      return "stopped";
    case "complete":
      return "too-late";
    default:
      return "unknown";
  }
}

/** A stop that landed before anything finished: nothing to report.
 *
 * A multi-unit job stopped in its first fold, trial, model or replicate has no
 * mean, ranking or best configuration to show, so the server ends the stream
 * normally and then fails the job with a message that says exactly that. It is
 * not a failure of the run: the researcher asked for the stop, and the stream
 * reached its end. A stream that never did is a crash and stays one. */
export function stoppedWithoutResult(args: {
  requested: boolean;
  failed: boolean;
  events: readonly TrainingEvent[];
}): boolean {
  return args.requested && args.failed && hasEnded(args.events);
}

/** The server's sentence without the exception class it puts in front
 *  (`RuntimeError: Parado antes…` reads as `Parado antes…`). */
export function plainReason(error: string): string {
  return error.replace(/^[A-Za-z]+(?:Error|Exception):\s*/, "");
}

/** What a stopped run has to say for itself, by the kind of stop it had. */
export type StopSummary =
  | { kind: "epoch"; epoch: number | null; total: number | null }
  | {
      kind: "units";
      unit: UnitKind;
      finished: number;
      stopped: number;
      planned: number | null;
    }
  | { kind: "phase"; bankKept: boolean };

/** The figures for the "stopped" line, and for how far the bar got.
 *
 * A multi-unit job reports how many units finished out of how many were planned
 * (and how many it cut, which are left out of its aggregate), read from its
 * report; without one, the epoch the stream reached stands in. A PatchCore
 * reports whether it got as far as saving its memory bank, which its `end`
 * event counts as one epoch (none when it stopped during extraction).
 */
export function stopSummary(
  mode: StopMode,
  events: readonly TrainingEvent[],
  report?: Record<string, unknown> | null,
): StopSummary {
  if (mode === "phase") {
    const { end } = milestones(events);
    return { kind: "phase", bankKept: (end?.total_epochs ?? 0) >= 1 };
  }
  if (mode === "trial" || mode === "fold" || mode === "model" || mode === "replicate") {
    const counts = unitCounts(report);
    if (counts !== null) {
      const { plannedTrials } = milestones(events);
      const fromReport = report?.["n_folds"];
      return {
        kind: "units",
        unit: mode,
        finished: counts.finished,
        stopped: counts.stopped,
        planned:
          plannedTrials > 0
            ? plannedTrials
            : typeof fromReport === "number"
              ? fromReport
              : null,
      };
    }
  }
  const reached = reachedEpoch(events);
  return { kind: "epoch", epoch: reached?.epoch ?? null, total: reached?.total ?? null };
}
