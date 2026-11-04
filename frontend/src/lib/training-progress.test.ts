import { describe, expect, it } from "vitest";
import type { TrainingEvent } from "../types/run";
import { liveBatchFraction } from "./training-progress";

const start: TrainingEvent = { event: "start", total_epochs: 4 };
const batch = (epoch: number, b: number, extra = {}): TrainingEvent => ({
  event: "batch",
  epoch,
  batch: b,
  total_batches: 10,
  total_epochs: 4,
  ...extra,
});
const epochEnd = (epoch: number): TrainingEvent => ({
  event: "epoch_end",
  epoch,
  total_epochs: 4,
  train_loss: 1,
  val_accuracy: 0.5,
});

describe("liveBatchFraction", () => {
  it("moves inside the first epoch, before any epoch_end", () => {
    // Epoch 1 of 4, batch 5 of 10: half of a quarter.
    expect(liveBatchFraction([start, batch(1, 5)])).toBeCloseTo(0.125);
  });

  it("counts the finished epochs before the one in flight", () => {
    const events = [start, epochEnd(1), epochEnd(2), batch(3, 2)];
    expect(liveBatchFraction(events)).toBeCloseTo((2 + 0.2) / 4);
  });

  it("yields to the epoch_end that follows it", () => {
    expect(liveBatchFraction([start, batch(1, 9), epochEnd(1)])).toBeNull();
  });

  it("is about the trial in flight only", () => {
    const trialStart: TrainingEvent = {
      event: "trial_start",
      trial_index: 1,
      total_trials: 2,
      overrides: {},
      seed: 0,
    };
    expect(liveBatchFraction([batch(4, 9, { trial_index: 0 }), trialStart])).toBeNull();
    expect(
      liveBatchFraction([trialStart, batch(1, 5, { trial_index: 1 })]),
    ).toBeCloseTo(0.125);
  });

  it("is null with no batch at all", () => {
    expect(liveBatchFraction([start, epochEnd(1)])).toBeNull();
  });
});
