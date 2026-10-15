/** What the "Como funciona" tab says is happening, read off the event stream
 * (ADR-118).
 *
 * Only two things on that tab are real: the macro *stage* of the run and the
 * *numbers* the trainer reported. Both come from here, from the same events the
 * log reads, and nothing is timed or invented. The particles that move across the
 * drawing are illustrative; they carry no information.
 *
 * The stage is what the stream proves:
 *
 * - `queued`: the server reported the run as waiting for the GPU;
 * - `preparing`: the run is going and has not reported an epoch (or, between the
 *   units of a multi-unit job, the next unit has not): data is loading and the
 *   model is being built;
 * - `training`: epochs are being reported, short of the last one — or, for a
 *   run that reports phases (PatchCore), a phase is under way;
 * - `testing`: the last planned epoch of the unit was reported and the unit has
 *   not closed, or the stream ended and the result is not out yet. The model is
 *   measured on the test split and the report is written;
 * - `done`, `interrupted`, `failed`: the run is over. A stop is not re-derived
 *   here: `stopped` is the verdict the sheet already reached with
 *   `stopOutcome` (lib/run-control.ts).
 *
 * Pure on purpose: no DOM and no fetch, so it is covered by plain vitest.
 */

import type { RunStatus, TrainingEvent } from "../types/run";
import { variesOnlyTheModel, type UnitKind } from "./run-control";
import type { MetricKind, StoryFamily, StorySpec } from "./story-family";
import { phaseKey, type PhaseKey } from "./training-phase";

/** The two tabs of the training sheet: the log (the default) and the story. */
export type SheetTab = "log" | "story";

export type StoryStage =
  | "queued"
  | "preparing"
  | "training"
  | "testing"
  | "done"
  | "interrupted"
  | "failed";

/** The stages in the order the strip on the tab shows them. The end states share
 *  the last slot. */
export const STAGE_ORDER: readonly StoryStage[] = [
  "preparing",
  "training",
  "testing",
  "done",
];

type EpochEnd = Extract<TrainingEvent, { event: "epoch_end" }>;
type PhaseEvent = Extract<TrainingEvent, { event: "phase" }>;

/** The unit of a multi-unit job that is running ("fold 2/5", "trial 3/10"). */
export interface StoryUnit {
  kind: UnitKind;
  /** 1-based. */
  index: number;
  total: number;
}

/** The phase a PatchCore run is in. `key` is `null` for a label this build does
 *  not know (shown as received). */
export interface StoryPhase {
  key: PhaseKey | null;
  label: string;
  done: number;
  total: number;
  /** `done / total`, clamped to 0..1. */
  fraction: number;
}

/** The live figures. Every one is `null` until the trainer has reported it. */
export interface StoryNumbers {
  epoch: number | null;
  totalEpochs: number | null;
  trainLoss: number | null;
  valLoss: number | null;
  /** The task's headline validation metric. */
  metric: { kind: MetricKind; value: number } | null;
}

export interface StoryState {
  stage: StoryStage;
  family: StoryFamily;
  unit: StoryUnit | null;
  numbers: StoryNumbers;
  /** Set for a run that reports phases, from the latest `phase` event. */
  phase: StoryPhase | null;
  /** How many epochs have been reported. It changes with every `epoch_end`, which
   *  is what makes the validation pulse fire once per epoch. */
  epochsSeen: number;
  /** The epoch the current unit started to overfit at, by `overfittingSince`. */
  overfittingSince: number | null;
}

export interface StoryInput {
  events: readonly TrainingEvent[];
  status: RunStatus["status"];
  /** The run ended because a stop cut it (`stopOutcome(...) === "stopped"`). */
  stopped: boolean;
  /** The unit a stop lands on (`stopMode`), which names the units of the job. */
  unitKind: UnitKind | null;
  spec: StorySpec;
}

/** The headline metric of an epoch report, or `null` when it does not carry it. */
function metricOf(e: EpochEnd, kind: MetricKind): number | null {
  const raw: unknown =
    kind === "map50"
      ? e.map50
      : kind === "r2"
        ? e["val_r2"]
        : kind === "miou"
          ? e["val_miou"]
          : kind === "auroc"
            ? e["val_auroc"]
            : e.val_accuracy;
  return typeof raw === "number" && Number.isFinite(raw) ? raw : null;
}

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function phaseOf(e: PhaseEvent): StoryPhase {
  const fraction = e.total > 0 ? Math.min(1, Math.max(0, e.done / e.total)) : 0;
  return { key: phaseKey(e.label), label: e.label, done: e.done, total: e.total, fraction };
}

/** How many consecutive rises of the validation loss make a sign of overfitting. */
export const OVERFIT_RISES = 3;

/** The epoch from which the losses show overfitting, or null (ADR-119).
 *
 * A sign, read from the numbers the trainer reported and nothing else: over the
 * last `OVERFIT_RISES` epochs the validation loss rose every epoch, and the
 * training loss ended lower than it started. One rise is noise, and a validation
 * loss that rises together with the training loss is a run that diverges, not one
 * that memorises. The answer is the first epoch of the climb, so the sentence can
 * say since when. A run without a validation loss never shows it.
 */
export function overfittingSince(epochs: readonly EpochEnd[]): number | null {
  if (epochs.length < OVERFIT_RISES + 1) return null;
  const window = epochs.slice(-(OVERFIT_RISES + 1));
  const val = window.map((e) => finite(e.val_loss));
  const train = window.map((e) => finite(e.train_loss));
  if (val.some((v) => v === null) || train.some((v) => v === null)) return null;
  for (let i = 1; i < window.length; i += 1) {
    if ((val[i] as number) <= (val[i - 1] as number)) return null;
  }
  if ((train[train.length - 1] as number) >= (train[0] as number)) return null;
  return window[1].epoch;
}

/** The story of a run: its stage, unit and numbers, from the events it streamed. */
export function trainingStory(input: StoryInput): StoryState {
  const { events, status, stopped, unitKind, spec } = input;

  let unitStart: Extract<TrainingEvent, { event: "start" }> | undefined;
  let lastEpoch: EpochEnd | undefined;
  // The epochs of the unit in flight: a sign of overfitting is one unit's.
  let unitEpochs: EpochEnd[] = [];
  let epochsSeen = 0;
  let trial: Extract<TrainingEvent, { event: "trial_start" }> | undefined;
  // The trial in `trial` has reported its `trial_end`.
  let trialClosed = false;
  let latestPhase: PhaseEvent | undefined;
  let ended = false;

  for (const e of events) {
    switch (e.event) {
      case "start":
        unitStart = e;
        break;
      case "trial_start":
        trial = e;
        trialClosed = false;
        // A new unit starts from nothing: the numbers of the last one are not its.
        unitStart = undefined;
        lastEpoch = undefined;
        unitEpochs = [];
        break;
      case "epoch_end":
        lastEpoch = e;
        unitEpochs.push(e);
        epochsSeen += 1;
        break;
      case "trial_end":
        trialClosed = true;
        break;
      case "phase":
        latestPhase = e;
        break;
      case "end":
        ended = true;
        break;
    }
  }

  const plannedUnits = trial?.total_trials ?? 0;
  const multiUnit = plannedUnits > 1;
  const lastUnit = trial !== undefined && trial.trial_index + 1 >= plannedUnits;
  // Between two units the next one is what is about to run.
  const upNext = multiUnit && trialClosed && !lastUnit;
  const unit: StoryUnit | null =
    multiUnit && trial !== undefined
      ? {
          // A sweep over the architecture alone compares models (ADR-059).
          kind: variesOnlyTheModel(trial.overrides) ? "model" : (unitKind ?? "trial"),
          index: Math.min(plannedUnits, trial.trial_index + 1 + (upNext ? 1 : 0)),
          total: plannedUnits,
        }
      : null;

  const total = finite(lastEpoch?.total_epochs) ?? finite(unitStart?.total_epochs);
  const numbers: StoryNumbers = {
    epoch: lastEpoch ? lastEpoch.epoch : null,
    totalEpochs: total !== null && total > 0 ? total : null,
    // PatchCore reports a loss of 0.0 because it trains nothing; it is not a
    // number to show.
    trainLoss: spec.family === "patchcore" || !lastEpoch ? null : finite(lastEpoch.train_loss),
    valLoss: lastEpoch ? finite(lastEpoch.val_loss) : null,
    metric: (() => {
      const value = lastEpoch ? metricOf(lastEpoch, spec.metric) : null;
      return value === null ? null : { kind: spec.metric, value };
    })(),
  };

  const phase = latestPhase ? phaseOf(latestPhase) : null;
  const phased = phase !== null;

  let stage: StoryStage;
  if (status === "failed") stage = "failed";
  else if (status === "completed") stage = stopped ? "interrupted" : "done";
  else if (status === "queued") stage = "queued";
  else if (ended) stage = "testing";
  else if (phased) stage = "training";
  else if (upNext) stage = "preparing";
  else if (multiUnit && trialClosed && lastUnit) stage = "testing";
  else if (lastEpoch === undefined) stage = "preparing";
  else if (numbers.totalEpochs !== null && lastEpoch.epoch >= numbers.totalEpochs) stage = "testing";
  else stage = "training";

  const overfitting =
    spec.family === "patchcore" || stage === "failed" || stage === "queued"
      ? null
      : overfittingSince(unitEpochs);

  return {
    stage,
    family: spec.family,
    unit,
    numbers,
    phase,
    epochsSeen,
    overfittingSince: overfitting,
  };
}

/** Which of the three flows the drawing carries in a stage. */
export interface StoryFlows {
  /** Data entering the first column. */
  data: boolean;
  /** The forward pass. */
  forward: boolean;
  /** The gradient coming back. */
  backward: boolean;
  /** The validation pass, in a colour of its own. */
  validation: boolean;
}

const NO_FLOWS: StoryFlows = { data: false, forward: false, backward: false, validation: false };

/** The flows a stage shows. Only running stages move anything.
 *
 * - preparing: data comes in and nothing has learned yet;
 * - training: data and the forward pass go in, the gradient comes back, and the
 *   validation pulse is armed for the end of each epoch;
 * - testing: only the evaluation pass runs;
 * - PatchCore has no gradient: it extracts and stores features (forward), and
 *   scores with the evaluation colour.
 */
export function activeFlows(
  stage: StoryStage,
  family: StoryFamily,
  phase: PhaseKey | null = null,
): StoryFlows {
  switch (stage) {
    case "preparing":
      return { ...NO_FLOWS, data: true };
    case "testing":
      return { ...NO_FLOWS, validation: true };
    case "training":
      if (family === "patchcore") {
        return phase === "scoring"
          ? { ...NO_FLOWS, validation: true }
          : { ...NO_FLOWS, data: true, forward: true };
      }
      return { data: true, forward: true, backward: true, validation: true };
    default:
      return NO_FLOWS;
  }
}

/** How the drawing moves.
 *
 * - `particles`: pulses travel along the curves;
 * - `arrows`: a still drawing with arrows on the curves, for a reader who asked the
 *   system for less motion (`prefers-reduced-motion: reduce`);
 * - `none`: nothing moves — the run is not going, or nobody is looking (the
 *   browser tab is hidden, or the sheet is minimized).
 */
export type StoryMotion = "particles" | "arrows" | "none";

export function storyMotion(args: {
  stage: StoryStage;
  reducedMotion: boolean;
  documentVisible: boolean;
  sheetVisible: boolean;
}): StoryMotion {
  const running =
    args.stage === "preparing" || args.stage === "training" || args.stage === "testing";
  if (!running) return "none";
  if (args.reducedMotion) return "arrows";
  return args.documentVisible && args.sheetVisible ? "particles" : "none";
}
