import { describe, expect, it } from "vitest";

import type { CodeNote, CodeSection } from "../api/client";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import type { TrainingEvent } from "../types/run";
import {
  FOLLOWING,
  followReducer,
  offPath,
  restingStep,
  RUN_STEPS,
  runPosition,
  sectionFor,
  stepView,
  stepWords,
  tourSteps,
  type FollowState,
  type RunStep,
} from "./run-steps";
import { storySpecFor } from "./story-family";
import { trainingStory, type StoryStage } from "./training-story";

const start: TrainingEvent = { event: "start", total_epochs: 3 };
const epoch = (n: number, total = 3): TrainingEvent => ({
  event: "epoch_end",
  epoch: n,
  total_epochs: total,
  train_loss: 0.5,
  val_loss: 0.6,
  val_accuracy: 0.8,
});
const batch = (epochN: number, b: number): TrainingEvent => ({
  event: "batch",
  epoch: epochN,
  batch: b,
  total_batches: 10,
  total_epochs: 3,
});
const phase = (label: string, done: number, total: number): TrainingEvent => ({
  event: "phase",
  label,
  done,
  total,
});

const classification = storySpecFor("classification", { model: { name: "resnet18" } });
const patchcore = storySpecFor("anomaly", { model: { name: "patchcore" } });

/** The position the tab reads for these events, through the story's own stage. */
function where(
  events: TrainingEvent[],
  status: "running" | "queued" | "completed" | "failed" = "running",
  spec = classification,
) {
  const story = trainingStory({ events, status, stopped: false, unitKind: null, spec });
  return runPosition(events, story.stage);
}

const note = (active: boolean | null): CodeNote => ({ first: 0, last: 0, text: "n", active });
const section = (id: string, step: RunStep, notes: CodeNote[] = []): CodeSection => ({
  id,
  step,
  summary: `${id} does things`,
  name: id,
  file: "visionforge/x.py",
  first_line: 1,
  lines: ["def f():"],
  notes,
  notes_resolved: notes.length,
  notes_total: notes.length,
});

const CLASSIFICATION_TOUR = RUN_STEPS.map((step) => section(step, step));
const ULTRALYTICS_TOUR = [
  section("yolo_args", "train"),
  section("yolo_augment", "data"),
  section("yolo_fit", "model"),
  section("yolo_epoch", "validate"),
  section("yolo_loop", "train"),
];
// A PatchCore run: the autoencoder's epoch is a branch it does not take.
const PATCHCORE_TOUR = [
  section("data", "data"),
  section("model", "model", [note(false), note(true)]),
  section("train", "train", [note(false), note(false)]),
  section("coreset", "train", [note(true)]),
  section("score", "validate", [note(false), note(true)]),
  section("threshold", "validate"),
  section("test", "test"),
];

describe("runPosition: what the stream proves", () => {
  it("runs nothing while queued, and nothing once the run is over", () => {
    expect(where([], "queued").steps).toEqual([]);
    expect(where([start, epoch(3), { event: "end", total_epochs: 3 }], "completed").steps).toEqual([]);
    expect(where([start, epoch(1)], "failed").steps).toEqual([]);
  });

  it("marks data, model and loss as one group before the trainer starts", () => {
    expect(where([])).toEqual({ steps: ["data", "model", "loss"], pins: [] });
  });

  it("marks training and validation as one group while an epoch has no batch report", () => {
    expect(where([start]).steps).toEqual(["train", "validate"]);
    expect(where([start, epoch(1)]).steps).toEqual(["train", "validate"]);
  });

  it("tells training apart once a batch is the newest word", () => {
    expect(where([start, batch(1, 4)]).steps).toEqual(["train"]);
    expect(where([start, epoch(1), batch(2, 3)]).steps).toEqual(["train"]);
    // The epoch that closes after the batch is in flight again, train or validate.
    expect(where([start, batch(1, 4), epoch(1)]).steps).toEqual(["train", "validate"]);
  });

  it("pins the function that walks the batches while an epoch is in flight", () => {
    expect(where([start, batch(1, 4)]).pins).toEqual(["train", "yolo_loop"]);
    expect(where([start]).pins).toEqual(["train", "yolo_loop"]);
  });

  it("moves to the test once the last planned epoch is reported, or the stream ends", () => {
    expect(where([start, epoch(3)])).toEqual({ steps: ["test"], pins: [] });
    expect(where([start, epoch(2), { event: "end", total_epochs: 2 }]).steps).toEqual(["test"]);
  });

  it("goes back to the group of preparation between two units of a job", () => {
    const events: TrainingEvent[] = [
      { event: "trial_start", trial_index: 0, total_trials: 3, overrides: {}, seed: 0 },
      start,
      epoch(3),
      { event: "trial_end", trial_index: 0, total_trials: 3, total_epochs: 3 },
    ];
    expect(where(events).steps).toEqual(["data", "model", "loss"]);
    const next: TrainingEvent[] = [
      ...events,
      { event: "trial_start", trial_index: 1, total_trials: 3, overrides: {}, seed: 1 },
    ];
    expect(where(next).steps).toEqual(["data", "model", "loss"]);
    expect(where([...next, start]).steps).toEqual(["train", "validate"]);
  });

  it("reads PatchCore's phases: extracting and the bank train, scoring validates", () => {
    const at = (label: string) => where([start, phase(label, 1, 4)], "running", patchcore);
    expect(at("extraindo features")).toEqual({ steps: ["train"], pins: [] });
    expect(at("montando o banco")).toEqual({ steps: ["train"], pins: ["coreset"] });
    expect(at("pontuando")).toEqual({ steps: ["validate"], pins: [] });
    // A label this build does not know proves only that it is fitting.
    expect(at("algo novo").steps).toEqual(["train", "validate"]);
  });

  it("moves PatchCore to the test once its single epoch is reported", () => {
    const events = [start, phase("pontuando", 2, 2), epoch(1, 1)];
    expect(where(events, "running", patchcore).steps).toEqual(["test"]);
  });
});

describe("tourSteps", () => {
  it("lists the steps a tour has code for, in run order", () => {
    expect(tourSteps(ULTRALYTICS_TOUR)).toEqual(["data", "model", "train", "validate"]);
    expect(tourSteps(PATCHCORE_TOUR)).toEqual(["data", "model", "train", "validate", "test"]);
  });

  it("keeps the six steps of a run when there is no tour", () => {
    expect(tourSteps(null)).toEqual(RUN_STEPS);
    expect(tourSteps([])).toEqual(RUN_STEPS);
  });
});

describe("sectionFor", () => {
  it("prefers the reader's pick, then a pinned function, then the first on this run's path", () => {
    expect(sectionFor(ULTRALYTICS_TOUR, "train", [], null)?.id).toBe("yolo_args");
    expect(sectionFor(ULTRALYTICS_TOUR, "train", ["train", "yolo_loop"], null)?.id).toBe("yolo_loop");
    expect(sectionFor(ULTRALYTICS_TOUR, "train", ["yolo_loop"], "yolo_args")?.id).toBe("yolo_args");
    expect(sectionFor(PATCHCORE_TOUR, "train", [], null)?.id).toBe("coreset");
  });

  it("has nothing to show for a step without code, or without a tour", () => {
    expect(sectionFor(ULTRALYTICS_TOUR, "test", [], null)).toBeNull();
    expect(sectionFor(null, "train", [], null)).toBeNull();
  });

  it("calls code off the path only when every note of it is a branch the run skips", () => {
    expect(offPath(section("a", "train", [note(false), note(false)]))).toBe(true);
    expect(offPath(section("a", "train", [note(false), note(null)]))).toBe(false);
    expect(offPath(section("a", "train"))).toBe(false);
  });
});

describe("following the run", () => {
  const view = (follow: FollowState, events: TrainingEvent[], sections = CLASSIFICATION_TOUR, stage?: StoryStage) => {
    const story = trainingStory({ events, status: "running", stopped: false, unitKind: null, spec: classification });
    const at = stage ?? story.stage;
    return stepView({ follow, position: runPosition(events, at), stage: at, sections });
  };

  it("shows the running step's code while following, and says it is live", () => {
    const v = view(FOLLOWING, [start, batch(1, 2)]);
    expect(v.step).toBe("train");
    expect(v.section?.id).toBe("train");
    expect(v.live).toBe(true);
    expect(v.running).toEqual(["train"]);
    expect(v.canResume).toBe(false);
  });

  it("shows the first step of a group, and marks the whole group as running", () => {
    const v = view(FOLLOWING, []);
    expect(v.running).toEqual(["data", "model", "loss"]);
    expect(v.step).toBe("data");
    expect(v.live).toBe(true);
  });

  it("shows Ultralytics' own loop while its epoch runs, not the arguments", () => {
    const v = view(FOLLOWING, [start, epoch(1)], ULTRALYTICS_TOUR);
    expect(v.running).toEqual(["train", "validate"]);
    expect(v.section?.id).toBe("yolo_loop");
  });

  it("pauses on a click and offers to resume while something runs", () => {
    const picked = followReducer(FOLLOWING, { kind: "step", step: "loss" });
    expect(picked.following).toBe(false);
    const v = view(picked, [start, batch(1, 2)]);
    expect(v.step).toBe("loss");
    expect(v.live).toBe(false);
    expect(v.canResume).toBe(true);
    // The run moves on; the reader's pick stays.
    expect(view(picked, [start, epoch(1), batch(2, 5)]).step).toBe("loss");
  });

  it("keeps a function picked inside a step", () => {
    const picked = followReducer(FOLLOWING, { kind: "section", step: "train", section: "yolo_args" });
    const v = view(picked, [start, batch(1, 2)], ULTRALYTICS_TOUR);
    expect(v.section?.id).toBe("yolo_args");
    expect(v.live).toBe(false);
  });

  it("resumes following the run, forgetting the pick", () => {
    const paused = followReducer(FOLLOWING, { kind: "step", step: "data" });
    const resumed = followReducer(paused, { kind: "resume" });
    expect(resumed).toEqual(FOLLOWING);
    expect(view(resumed, [start, batch(1, 2)]).step).toBe("train");
  });

  it("marks nothing as running once the run is over, and lets the reader browse", () => {
    const events = [start, epoch(3), { event: "end", total_epochs: 3 } as TrainingEvent];
    const done = view(FOLLOWING, events, CLASSIFICATION_TOUR, "done");
    expect(done.running).toEqual([]);
    expect(done.live).toBe(false);
    expect(done.canResume).toBe(false);
    // It rests on the test, the last step that ran, rather than jumping back.
    expect(done.step).toBe("test");
    const browsing = followReducer(FOLLOWING, { kind: "step", step: "model" });
    expect(view(browsing, events, CLASSIFICATION_TOUR, "done").step).toBe("model");
  });

  it("does not mark a running step the tour has no code for", () => {
    // Ultralytics' tour has no test step.
    const v = view(FOLLOWING, [start, epoch(3)], ULTRALYTICS_TOUR);
    expect(v.running).toEqual([]);
    expect(v.live).toBe(false);
    expect(v.step).toBe("validate");
  });
});

describe("restingStep", () => {
  it("rests on the training before the run reaches the test, and on the test after", () => {
    expect(restingStep("queued", RUN_STEPS)).toBe("train");
    expect(restingStep("failed", RUN_STEPS)).toBe("train");
    expect(restingStep("done", RUN_STEPS)).toBe("test");
    expect(restingStep("interrupted", ["data", "train", "validate"])).toBe("validate");
  });
});

describe("stepWords", () => {
  it("words PatchCore's training and validation as what it does, and the rest as any run", () => {
    expect(stepWords(pt.codeTour, "train", true)).toBe(pt.codeTour.patchcoreSteps.train);
    expect(stepWords(pt.codeTour, "validate", true)).toBe(pt.codeTour.patchcoreSteps.validate);
    expect(stepWords(pt.codeTour, "data", true)).toBe(pt.codeTour.steps.data);
    expect(stepWords(en.codeTour, "train", false)).toBe(en.codeTour.steps.train);
  });
});
