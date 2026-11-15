/** Where the run is, as steps, and which code the "Como funciona" tab shows.
 *
 * The tab lists the run as steps (data, model, loss, train, validate, test) and
 * shows the real code of one of them. While the run is going it follows the step
 * the event stream proves is running; a click pauses that, and "voltar ao que
 * está rodando" resumes it.
 *
 * Honesty, as in ADR-118: the position comes only from what the stream proves,
 * and its grain is the step, never a line. A loop passes its lines thousands of
 * times a second and no tracer runs. When the stream cannot tell two steps apart
 * (data, model and loss while the run prepares; training and validation inside an
 * epoch whose trainer sent no `batch` yet), the group is what runs, and the tab
 * says so instead of picking one. The epoch's last `batch` is always sent
 * (ADR-205), so validation is told apart once training is over.
 *
 * Pure on purpose: no DOM and no fetch, so it is covered by plain vitest.
 */

import type { CodeSection } from "../api/client";
import type { Dict } from "../i18n/pt";
import type { TrainingEvent } from "../types/run";
import { phaseKey } from "./training-phase";
import type { StoryStage } from "./training-story";

/** The steps of a run, as `code_tour.STEPS` declares them on the server. */
export type RunStep = CodeSection["step"];

/** The steps in the order the tab lists them. */
export const RUN_STEPS: readonly RunStep[] = ["data", "model", "loss", "train", "validate", "test"];

/** Before the trainer's `start`: the data, the model and the loss are being built,
 *  and the stream has no event that tells them apart. */
const PREPARING: readonly RunStep[] = ["data", "model", "loss"];
/** An epoch in flight with no `batch` after its start: it trains, then validates. */
const IN_EPOCH: readonly RunStep[] = ["train", "validate"];

/** The function that walks the batches: `train` in the loops VisionForge owns,
 *  `yolo_loop` in Ultralytics' (where the validation of the epoch runs too). A
 *  server test pins that every tour's train step has one of them. */
export const LOOP_SECTIONS: readonly string[] = ["train", "yolo_loop"];

/** PatchCore's memory bank: its `phase` "montando o banco" is this function at work. */
export const BANK_SECTION = "coreset";

/** What the stream proves about where the run is. */
export interface RunPosition {
  /** One of these steps is running: one step when the stream tells, a group when
   *  it cannot, none when nothing runs. */
  steps: readonly RunStep[];
  /** Ids of the functions the stream shows at work, by preference. Empty when it
   *  proves no particular function. */
  pins: readonly string[];
}

export const NOWHERE: RunPosition = { steps: [], pins: [] };

type UnitEvent = Extract<
  TrainingEvent,
  { event: "start" | "trial_start" | "trial_end" | "epoch_end" | "batch" | "phase" }
>;

function isUnitEvent(e: TrainingEvent): e is UnitEvent {
  return (
    e.event === "start" ||
    e.event === "trial_start" ||
    e.event === "trial_end" ||
    e.event === "epoch_end" ||
    e.event === "batch" ||
    e.event === "phase"
  );
}

/** The latest event that says where the unit in flight is. */
function latestUnitEvent(events: readonly TrainingEvent[]): UnitEvent | undefined {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const e = events[i];
    if (isUnitEvent(e)) return e;
  }
  return undefined;
}

function phasePosition(label: string): RunPosition {
  switch (phaseKey(label)) {
    case "extractingFeatures":
      return { steps: ["train"], pins: [] };
    case "buildingBank":
      return { steps: ["train"], pins: [BANK_SECTION] };
    case "scoring":
      return { steps: ["validate"], pins: [] };
    default:
      // A phase this build does not know: it is fitting, not more.
      return { steps: IN_EPOCH, pins: [] };
  }
}

/** Where the run is, from its stage (`trainingStory`) and its events.
 *
 * - queued, finished, failed: nothing runs;
 * - `testing` (the last planned epoch was reported, or the stream ended): the test
 *   and the report;
 * - nothing yet, or a unit opened or closed with the next one not started: the
 *   data, the model and the loss, as a group;
 * - after `start`, or after an epoch that is not the last: an epoch in flight,
 *   training or validation, as a group, in the function that walks the batches;
 * - a `batch` is the newest word: training, in that function; the epoch's last
 *   batch (`batch >= total_batches`) means its validation, alone;
 * - a PatchCore `phase`: extracting features and building the bank are its
 *   training (the bank pins its function), scoring is its validation.
 */
export function runPosition(events: readonly TrainingEvent[], stage: StoryStage): RunPosition {
  if (stage === "testing") return { steps: ["test"], pins: [] };
  if (stage !== "preparing" && stage !== "training") return NOWHERE;
  const latest = latestUnitEvent(events);
  switch (latest?.event) {
    case "start":
      return { steps: IN_EPOCH, pins: LOOP_SECTIONS };
    case "batch":
      // The trainer always reports an epoch's last batch (ADR-205): past it the
      // epoch validates. No pin: the stream proves no particular function there.
      return latest.batch >= latest.total_batches
        ? { steps: ["validate"], pins: [] }
        : { steps: ["train"], pins: LOOP_SECTIONS };
    case "epoch_end":
      return latest.total_epochs > 0 && latest.epoch >= latest.total_epochs
        ? { steps: ["test"], pins: [] }
        : { steps: IN_EPOCH, pins: LOOP_SECTIONS };
    case "phase":
      return phasePosition(latest.label);
    default:
      return { steps: PREPARING, pins: [] };
  }
}

/** The steps a tour has code for, in run order. Without a tour (still loading, or
 *  a task that has none) the run is still the six steps. */
export function tourSteps(sections: readonly CodeSection[] | null): RunStep[] {
  if (!sections || sections.length === 0) return [...RUN_STEPS];
  return RUN_STEPS.filter((step) => sections.some((s) => s.step === step));
}

/** Code this run does not take: every note of the function is a branch it skips
 *  (the autoencoder's epoch in a PatchCore run). */
export function offPath(section: CodeSection): boolean {
  return section.notes.length > 0 && section.notes.every((n) => n.active === false);
}

/** Whether the tab follows the run, and what the reader picked when it does not. */
export interface FollowState {
  following: boolean;
  step: RunStep | null;
  /** A function picked inside the step; null for the step's own default. */
  section: string | null;
}

export const FOLLOWING: FollowState = { following: true, step: null, section: null };

export type FollowAction =
  | { kind: "step"; step: RunStep }
  | { kind: "section"; step: RunStep; section: string }
  | { kind: "resume" };

/** A click on a step or a function pauses following; resuming forgets the pick. */
export function followReducer(_state: FollowState, action: FollowAction): FollowState {
  switch (action.kind) {
    case "step":
      return { following: false, step: action.step, section: null };
    case "section":
      return { following: false, step: action.step, section: action.section };
    case "resume":
      return FOLLOWING;
  }
}

/** The step shown when nothing running decides it: the test once the run has
 *  reached it, the training before. A tour without that step falls back to its
 *  last step (after training) or its first (before). */
export function restingStep(stage: StoryStage, steps: readonly RunStep[]): RunStep {
  const late = stage === "testing" || stage === "done" || stage === "interrupted";
  const preferred: RunStep = late ? "test" : "train";
  if (steps.includes(preferred)) return preferred;
  return (late ? steps[steps.length - 1] : steps[0]) ?? preferred;
}

/** The function shown for `step`: the reader's pick, else one the stream pins,
 *  else the first on this run's path, else the first. Null when the step has no
 *  code in this tour. */
export function sectionFor(
  sections: readonly CodeSection[] | null,
  step: RunStep,
  pins: readonly string[],
  picked: string | null,
): CodeSection | null {
  const inStep = (sections ?? []).filter((s) => s.step === step);
  if (inStep.length === 0) return null;
  const byId = (id: string) => inStep.find((s) => s.id === id);
  const chosen = (picked ? byId(picked) : undefined) ?? pins.map(byId).find((s) => s !== undefined);
  return chosen ?? inStep.find((s) => !offPath(s)) ?? inStep[0];
}

export interface StepView {
  /** The steps this tour lists. */
  steps: RunStep[];
  /** The listed steps one of which is running (a group when the stream cannot tell). */
  running: RunStep[];
  /** The step whose code is on screen. */
  step: RunStep;
  /** Its function on screen; null when the step has no code (or there is no tour). */
  section: CodeSection | null;
  /** The code on screen is there because the tab follows the run: it is the
   *  running step's. */
  live: boolean;
  /** Something runs and the tab is not following it: it can be resumed. */
  canResume: boolean;
}

/** What the tab shows, from the reader's choice, the run's position and the tour. */
export function stepView(args: {
  follow: FollowState;
  position: RunPosition;
  stage: StoryStage;
  sections: readonly CodeSection[] | null;
}): StepView {
  const { follow, position, stage, sections } = args;
  const steps = tourSteps(sections);
  const running = position.steps.filter((s) => steps.includes(s));
  const live = follow.following && running.length > 0;
  const step = live
    ? running[0]
    : !follow.following && follow.step !== null && steps.includes(follow.step)
      ? follow.step
      : restingStep(stage, steps);
  const section = live
    ? sectionFor(sections, step, position.pins, null)
    : sectionFor(sections, step, [], follow.section);
  return { steps, running, step, section, live, canResume: !follow.following && running.length > 0 };
}

/** The title and sentence of a step. PatchCore's training and validation are
 *  worded as what it does instead: it has no epochs and no gradient. */
export function stepWords(
  words: Dict["codeTour"],
  step: RunStep,
  patchcore: boolean,
): { title: string; text: string } {
  if (patchcore && (step === "train" || step === "validate")) return words.patchcoreSteps[step];
  return words.steps[step];
}
