import { describe, expect, it } from "vitest";

import type { TrainingEvent } from "../types/run";
import { GENERIC_STORY, storySpecFor, type StorySpec } from "./story-family";
import {
  activeFlows,
  overfittingSince,
  STAGE_ORDER,
  storyMotion,
  trainingStory,
  type StoryInput,
  type StoryStage,
} from "./training-story";

const start = (total: number): TrainingEvent => ({ event: "start", total_epochs: total });
const end = (total: number, trials?: number): TrainingEvent => ({
  event: "end",
  total_epochs: total,
  total_trials: trials,
});
const epoch = (
  n: number,
  total: number,
  extra: Partial<Extract<TrainingEvent, { event: "epoch_end" }>> = {},
): TrainingEvent => ({
  event: "epoch_end",
  epoch: n,
  total_epochs: total,
  train_loss: 1 / n,
  val_loss: 1.2 / n,
  val_accuracy: 0.5 + n / 20,
  ...extra,
});
const trialStart = (index: number, of: number): TrainingEvent => ({
  event: "trial_start",
  trial_index: index,
  total_trials: of,
  overrides: {},
  seed: 0,
});
const trialEnd = (index: number, of: number, epochs = 2): TrainingEvent => ({
  event: "trial_end",
  trial_index: index,
  total_trials: of,
  total_epochs: epochs,
});
const phase = (label: string, done: number, total: number): TrainingEvent => ({
  event: "phase",
  label,
  done,
  total,
});

const classification: StorySpec = storySpecFor("classification", { model: { name: "resnet18" } });

/** The story of a run, with the defaults of an ordinary running classification. */
const story = (events: TrainingEvent[], over: Partial<StoryInput> = {}) =>
  trainingStory({
    events,
    status: "running",
    stopped: false,
    unitKind: null,
    spec: classification,
    ...over,
  });

describe("trainingStory: a single run", () => {
  it("is preparing until the first epoch is reported", () => {
    expect(story([]).stage).toBe("preparing");
    const s = story([start(3)]);
    expect(s.stage).toBe("preparing");
    // The planned length is known from `start`, the epoch is not.
    expect(s.numbers).toMatchObject({ epoch: null, totalEpochs: 3, trainLoss: null });
    expect(s.unit).toBeNull();
    expect(s.phase).toBeNull();
  });

  it("trains while epochs short of the last are reported, with their numbers", () => {
    const s = story([start(3), epoch(1, 3), epoch(2, 3)]);
    expect(s.stage).toBe("training");
    expect(s.epochsSeen).toBe(2);
    expect(s.numbers.epoch).toBe(2);
    expect(s.numbers.totalEpochs).toBe(3);
    expect(s.numbers.trainLoss).toBeCloseTo(0.5);
    expect(s.numbers.valLoss).toBeCloseTo(0.6);
    expect(s.numbers.metric).toEqual({ kind: "accuracy", value: 0.6 });
  });

  it("is evaluating on the test split after the last planned epoch, until the end", () => {
    expect(story([start(3), epoch(1, 3), epoch(2, 3), epoch(3, 3)]).stage).toBe("testing");
    // The stream ended but the result is not out yet: still the evaluation.
    expect(story([start(3), epoch(3, 3), end(3)]).stage).toBe("testing");
  });

  it("is done when the run completes, whatever the stream said", () => {
    expect(story([start(3), epoch(3, 3), end(3)], { status: "completed" }).stage).toBe("done");
    // Early stopping on its own ends short of the plan and is still done.
    expect(story([start(10), epoch(4, 10), end(4)], { status: "completed" }).stage).toBe("done");
    // A stop that cut nothing (the sheet's "too late") is done as well.
    expect(
      story([start(3), epoch(3, 3), end(3)], { status: "completed", stopped: false }).stage,
    ).toBe("done");
  });

  it("is interrupted when the sheet's stop verdict says a stop cut it", () => {
    const s = story([start(5), epoch(1, 5), epoch(2, 5), end(2)], {
      status: "completed",
      stopped: true,
    });
    expect(s.stage).toBe("interrupted");
    expect(s.numbers.epoch).toBe(2);
  });

  it("is interrupted when stopped before the first epoch, with no epoch to show", () => {
    const s = story([start(5), end(0)], { status: "completed", stopped: true });
    expect(s.stage).toBe("interrupted");
    expect(s.numbers.epoch).toBeNull();
    expect(s.numbers.totalEpochs).toBe(5);
    expect(s.numbers.metric).toBeNull();
  });

  it("keeps training while a stop is only requested: the verdict comes with the end", () => {
    // `stopped` is false until the run has ended, so a pending stop changes nothing.
    expect(story([start(5), epoch(2, 5)], { stopped: false }).stage).toBe("training");
  });

  it("is failed whenever the run failed, even before a single event", () => {
    expect(story([], { status: "failed" }).stage).toBe("failed");
    expect(story([start(3), epoch(1, 3)], { status: "failed" }).stage).toBe("failed");
    // A failure is never an interruption, whatever the stop verdict claims.
    expect(story([start(3)], { status: "failed", stopped: true }).stage).toBe("failed");
  });

  it("waits in the queue until the server starts the run", () => {
    expect(story([], { status: "queued" }).stage).toBe("queued");
  });

  it("reads the planned length from the epoch once there is one", () => {
    const s = story([start(3), epoch(1, 8)]);
    expect(s.numbers.totalEpochs).toBe(8);
  });

  it("ignores a planned length of zero", () => {
    expect(story([start(0)]).numbers.totalEpochs).toBeNull();
  });

  it("counts every epoch report, which is what fires the validation pulse", () => {
    const events = [start(4), epoch(1, 4), epoch(2, 4)];
    expect(story(events).epochsSeen).toBe(2);
    expect(story([...events, epoch(3, 4)]).epochsSeen).toBe(3);
  });
});

describe("trainingStory: the headline metric", () => {
  it("reads the metric of each task from its own field", () => {
    const cases: Array<[string, Partial<Extract<TrainingEvent, { event: "epoch_end" }>>, string, number]> = [
      ["detection", { map50: 0.42 }, "map50", 0.42],
      ["regression", { val_r2: 0.81 }, "r2", 0.81],
      ["segmentation", { val_miou: 0.67 }, "miou", 0.67],
      ["anomaly", { val_auroc: 0.93 }, "auroc", 0.93],
    ];
    for (const [task, extra, kind, value] of cases) {
      const s = story([start(2), epoch(1, 2, extra)], { spec: storySpecFor(task, {}) });
      expect(s.numbers.metric, task).toEqual({ kind, value });
    }
  });

  it("shows a detection run with no mAP yet as having none, not as zero", () => {
    const s = story([start(2), epoch(1, 2, { map50: null })], { spec: storySpecFor("detection", {}) });
    expect(s.numbers.metric).toBeNull();
  });

  it("reads a researcher's own task from the compat field", () => {
    const s = story([start(2), epoch(1, 2, { val_accuracy: 0.9 })], {
      spec: GENERIC_STORY,
    });
    expect(s.numbers.metric).toEqual({ kind: "primary", value: 0.9 });
  });

  it("leaves out a loss the task does not report", () => {
    const s = story([start(2), { ...epoch(1, 2), val_loss: undefined } as TrainingEvent]);
    expect(s.numbers.valLoss).toBeNull();
    expect(s.numbers.trainLoss).not.toBeNull();
  });
});

describe("trainingStory: multi-unit jobs", () => {
  const kfold = (extra: Partial<StoryInput> = {}) => ({ unitKind: "fold" as const, ...extra });

  it("names the unit that is running", () => {
    const s = story([trialStart(0, 3)], kfold());
    expect(s.unit).toEqual({ kind: "fold", index: 1, total: 3 });
    expect(s.stage).toBe("preparing");
  });

  it("walks a K-fold through each fold's stages", () => {
    const events: TrainingEvent[] = [trialStart(0, 3), start(2), epoch(1, 2, { trial_index: 0, total_trials: 3 })];
    expect(story(events, kfold()).stage).toBe("training");
    events.push(epoch(2, 2, { trial_index: 0, total_trials: 3 }));
    // The last epoch of the fold is in; the fold has not closed.
    expect(story(events, kfold()).stage).toBe("testing");
    events.push(trialEnd(0, 3));
    // Between folds: the next one is about to start, and is the one named.
    const between = story(events, kfold());
    expect(between.stage).toBe("preparing");
    expect(between.unit).toEqual({ kind: "fold", index: 2, total: 3 });
    events.push(trialStart(1, 3));
    const second = story(events, kfold());
    expect(second.unit).toEqual({ kind: "fold", index: 2, total: 3 });
    expect(second.stage).toBe("preparing");
    // The new fold starts from nothing: the last one's figures are not its own.
    expect(second.numbers.epoch).toBeNull();
    expect(second.numbers.metric).toBeNull();
  });

  it("is evaluating after the last unit closes, until the end, then done", () => {
    const events: TrainingEvent[] = [
      trialStart(0, 2),
      epoch(2, 2, { trial_index: 0 }),
      trialEnd(0, 2),
      trialStart(1, 2),
      epoch(2, 2, { trial_index: 1 }),
      trialEnd(1, 2),
    ];
    const closed = story(events, kfold());
    expect(closed.stage).toBe("testing");
    // The last unit stays the one named: there is no third.
    expect(closed.unit).toEqual({ kind: "fold", index: 2, total: 2 });
    expect(story([...events, end(0, 2)], kfold()).stage).toBe("testing");
    expect(story([...events, end(0, 2)], kfold({ status: "completed" })).stage).toBe("done");
  });

  it("calls a search's units trials and a replicate set's units replicates", () => {
    expect(story([trialStart(2, 10)], { unitKind: "trial" }).unit).toEqual({
      kind: "trial",
      index: 3,
      total: 10,
    });
    expect(story([trialStart(0, 4)], { unitKind: "replicate" }).unit?.kind).toBe("replicate");
    // Without a stop point to name the job, a unit is a trial.
    expect(story([trialStart(0, 4)], { unitKind: null }).unit?.kind).toBe("trial");
  });

  it("shows no unit for a job of one", () => {
    expect(story([trialStart(0, 1)]).unit).toBeNull();
  });

  it("is interrupted when a stop cut a unit", () => {
    const events: TrainingEvent[] = [
      trialStart(0, 3),
      epoch(2, 2, { trial_index: 0 }),
      trialEnd(0, 3),
      trialStart(1, 3),
      end(0, 2),
    ];
    const s = story(events, kfold({ status: "completed", stopped: true }));
    expect(s.stage).toBe("interrupted");
    expect(s.unit).toEqual({ kind: "fold", index: 2, total: 3 });
  });
});

describe("trainingStory: PatchCore phases", () => {
  const patchcore = storySpecFor("anomaly", { model: { name: "patchcore" } });
  const pc = (events: TrainingEvent[], over: Partial<StoryInput> = {}) =>
    story(events, { spec: patchcore, ...over });

  it("is preparing until the first phase is reported", () => {
    expect(pc([start(1)]).stage).toBe("preparing");
    expect(pc([start(1)]).phase).toBeNull();
  });

  it("follows the three phases, with the progress of the current one", () => {
    const extracting = pc([start(1), phase("extraindo features", 3, 12)]);
    expect(extracting.stage).toBe("training");
    expect(extracting.phase).toEqual({
      key: "extractingFeatures",
      label: "extraindo features",
      done: 3,
      total: 12,
      fraction: 0.25,
    });

    const bank = pc([start(1), phase("extraindo features", 12, 12), phase("montando o banco", 50, 100)]);
    expect(bank.phase?.key).toBe("buildingBank");
    expect(bank.phase?.fraction).toBe(0.5);

    const scoring = pc([start(1), phase("pontuando", 1, 2)]);
    expect(scoring.phase?.key).toBe("scoring");
    expect(scoring.stage).toBe("training");
  });

  it("stays on the phases when the one epoch report arrives, and hides the loss it zeroes", () => {
    const s = pc([
      start(1),
      phase("pontuando", 2, 2),
      epoch(1, 1, { train_loss: 0, val_auroc: 0.93 }),
    ]);
    // An epoch 1 of 1 would read as the end of training in a run with epochs.
    expect(s.stage).toBe("training");
    expect(s.numbers.trainLoss).toBeNull();
    expect(s.numbers.metric).toEqual({ kind: "auroc", value: 0.93 });
  });

  it("is evaluating once the stream ends, then done or interrupted", () => {
    const events = [start(1), phase("pontuando", 2, 2), epoch(1, 1), end(1)];
    expect(pc(events).stage).toBe("testing");
    expect(pc(events, { status: "completed" }).stage).toBe("done");
    // Stopped during extraction: no epoch, `end` of zero.
    expect(
      pc([start(1), phase("extraindo features", 1, 9), end(0)], {
        status: "completed",
        stopped: true,
      }).stage,
    ).toBe("interrupted");
  });

  it("passes on a label it does not know, as received", () => {
    const s = pc([start(1), phase("outra coisa", 1, 4)]);
    expect(s.phase).toMatchObject({ key: null, label: "outra coisa", fraction: 0.25 });
  });

  it("does not divide by a zero total", () => {
    expect(pc([phase("pontuando", 0, 0)]).phase?.fraction).toBe(0);
  });
});

describe("activeFlows", () => {
  it("moves nothing when the run is not going", () => {
    for (const stage of ["queued", "done", "interrupted", "failed"] as StoryStage[]) {
      expect(activeFlows(stage, "cnn"), stage).toEqual({
        data: false,
        forward: false,
        backward: false,
        validation: false,
      });
    }
  });

  it("brings data in while preparing, and nothing learns yet", () => {
    expect(activeFlows("preparing", "cnn")).toEqual({
      data: true,
      forward: false,
      backward: false,
      validation: false,
    });
  });

  it("runs the three flows while training", () => {
    for (const family of ["cnn", "transformer", "encoder_decoder", "detector", "autoencoder"] as const) {
      expect(activeFlows("training", family), family).toEqual({
        data: true,
        forward: true,
        backward: true,
        validation: true,
      });
    }
  });

  it("runs only the evaluation pass while testing", () => {
    expect(activeFlows("testing", "cnn")).toEqual({
      data: false,
      forward: false,
      backward: false,
      validation: true,
    });
  });

  it("gives PatchCore no gradient, and scores with the evaluation colour", () => {
    expect(activeFlows("training", "patchcore", "extractingFeatures").backward).toBe(false);
    expect(activeFlows("training", "patchcore", "buildingBank")).toMatchObject({
      forward: true,
      validation: false,
    });
    expect(activeFlows("training", "patchcore", "scoring")).toMatchObject({
      forward: false,
      validation: true,
    });
  });
});

describe("storyMotion", () => {
  const visible = { reducedMotion: false, documentVisible: true, sheetVisible: true };

  it("sends particles through a run that is going and being watched", () => {
    for (const stage of ["preparing", "training", "testing"] as StoryStage[]) {
      expect(storyMotion({ stage, ...visible }), stage).toBe("particles");
    }
  });

  it("draws arrows instead of particles for a reader who asked for less motion", () => {
    for (const stage of ["preparing", "training", "testing"] as StoryStage[]) {
      expect(storyMotion({ stage, ...visible, reducedMotion: true }), stage).toBe("arrows");
    }
    // Even with the tab in the background: arrows are still, they cost nothing.
    expect(
      storyMotion({ stage: "training", ...visible, reducedMotion: true, documentVisible: false }),
    ).toBe("arrows");
  });

  it("pauses when the browser tab is hidden or the sheet is minimized", () => {
    expect(storyMotion({ stage: "training", ...visible, documentVisible: false })).toBe("none");
    expect(storyMotion({ stage: "training", ...visible, sheetVisible: false })).toBe("none");
  });

  it("does not move a run that is not going, whatever the preference", () => {
    for (const stage of ["queued", "done", "interrupted", "failed"] as StoryStage[]) {
      expect(storyMotion({ stage, ...visible }), stage).toBe("none");
      expect(storyMotion({ stage, ...visible, reducedMotion: true }), stage).toBe("none");
    }
  });
});

describe("STAGE_ORDER", () => {
  it("lists the stages of a run in the order it goes through them", () => {
    expect(STAGE_ORDER).toEqual(["preparing", "training", "testing", "done"]);
  });
});

describe("overfittingSince", () => {
  type EpochEnd = Extract<TrainingEvent, { event: "epoch_end" }>;
  /** Epochs 1..n with the given losses. */
  const series = (train: number[], val: number[]): EpochEnd[] =>
    train.map((tl, i) => ({
      event: "epoch_end",
      epoch: i + 1,
      total_epochs: 20,
      train_loss: tl,
      val_loss: val[i],
      val_accuracy: 0.5,
    }));

  it("names the first epoch of three straight rises while training falls", () => {
    const epochs = series([1.0, 0.8, 0.6, 0.5, 0.4, 0.3], [1.0, 0.7, 0.6, 0.65, 0.7, 0.8]);
    expect(overfittingSince(epochs)).toBe(4);
  });

  it("does not call two rises a sign", () => {
    const epochs = series([1.0, 0.8, 0.6, 0.5, 0.4], [1.0, 0.7, 0.6, 0.65, 0.7]);
    expect(overfittingSince(epochs)).toBeNull();
  });

  it("does not call a dip in the climb a sign", () => {
    const epochs = series([1.0, 0.8, 0.6, 0.5, 0.4], [0.6, 0.65, 0.64, 0.7, 0.8]);
    expect(overfittingSince(epochs)).toBeNull();
  });

  it("does not call both losses rising a sign: that run diverges", () => {
    const epochs = series([0.5, 0.6, 0.7, 0.8], [0.6, 0.7, 0.8, 0.9]);
    expect(overfittingSince(epochs)).toBeNull();
  });

  it("stays silent without a validation loss", () => {
    const epochs = series([1.0, 0.8, 0.6, 0.5], [Number.NaN, Number.NaN, Number.NaN, Number.NaN]);
    expect(overfittingSince(epochs)).toBeNull();
    expect(overfittingSince([])).toBeNull();
  });
});

describe("trainingStory: overfitting", () => {
  const rising: TrainingEvent[] = [
    start(10),
    epoch(1, 10, { train_loss: 1.0, val_loss: 0.6 }),
    epoch(2, 10, { train_loss: 0.8, val_loss: 0.65 }),
    epoch(3, 10, { train_loss: 0.6, val_loss: 0.7 }),
    epoch(4, 10, { train_loss: 0.5, val_loss: 0.8 }),
  ];

  it("reports the sign while the run trains", () => {
    expect(story(rising).overfittingSince).toBe(2);
  });

  it("keeps it on the finished run, where the reader looks back", () => {
    expect(story(rising, { status: "completed" }).overfittingSince).toBe(2);
  });

  it("drops it on a failed run, whose numbers are not the point", () => {
    expect(story(rising, { status: "failed" }).overfittingSince).toBeNull();
  });

  it("reads one unit at a time: the next unit starts clean", () => {
    const events: TrainingEvent[] = [
      trialStart(0, 2),
      ...rising.slice(1),
      trialEnd(0, 2, 4),
      trialStart(1, 2),
      epoch(1, 10, { train_loss: 1.0, val_loss: 0.9 }),
    ];
    expect(story(events).overfittingSince).toBeNull();
  });

  it("never shows on PatchCore, which has no losses to read", () => {
    const patchcore = storySpecFor("anomaly", { model: { name: "patchcore" } });
    expect(story(rising, { spec: patchcore }).overfittingSince).toBeNull();
  });
});

describe("trainingStory: a sweep over the architecture", () => {
  it("names its units as models", () => {
    const events: TrainingEvent[] = [
      { event: "trial_start", trial_index: 1, total_trials: 3, overrides: { "model.name": "resnet34" }, seed: 0 },
      epoch(1, 10),
    ];
    expect(story(events).unit).toEqual({ kind: "model", index: 2, total: 3 });
  });

  it("keeps a sweep over anything else a trial", () => {
    const events: TrainingEvent[] = [
      {
        event: "trial_start",
        trial_index: 0,
        total_trials: 3,
        overrides: { "model.name": "resnet34", "training.learning_rate": 0.01 },
        seed: 0,
      },
      epoch(1, 10),
    ];
    expect(story(events).unit?.kind).toBe("trial");
  });
});
