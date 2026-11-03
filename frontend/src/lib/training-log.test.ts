import { describe, expect, it } from "vitest";
import { pt } from "../i18n/pt";
import type { TrainingEvent } from "../types/run";
import { logLinesFor } from "./training-log";

const epoch = (n: number): TrainingEvent => ({
  event: "epoch_end",
  epoch: n,
  total_epochs: 3,
  train_loss: 0.5,
  val_loss: 0.25,
  val_accuracy: 0.75,
});

describe("logLinesFor", () => {
  it("keeps every epoch of a replay that lands at once", () => {
    const { lines } = logLinesFor([epoch(1), epoch(2), epoch(3)], null);
    expect(lines).toEqual([
      "> epoch 1/3 · loss=0.5000 · val_loss=0.2500 · val_acc=0.7500",
      "> epoch 2/3 · loss=0.5000 · val_loss=0.2500 · val_acc=0.7500",
      "> epoch 3/3 · loss=0.5000 · val_loss=0.2500 · val_acc=0.7500",
    ]);
  });

  it("puts a trial's banner before its epochs", () => {
    const { lines } = logLinesFor(
      [
        { event: "trial_start", trial_index: 1, total_trials: 2, overrides: { "model.name": "a" }, seed: 0 },
        epoch(1),
      ],
      null,
    );
    expect(lines[0]).toBe("── model 2/2 · a ──");
    expect(lines[1]).toContain("epoch 1/3");
  });

  it("writes one line per phase and remembers it across calls", () => {
    const update = (done: number): TrainingEvent => ({
      event: "phase",
      label: "bank",
      done,
      total: 100,
    });
    const first = logLinesFor([update(1), update(2)], null);
    expect(first.lines).toHaveLength(1);
    expect(typeof first.lines[0] === "function" && first.lines[0](pt)).toContain("…");
    expect(logLinesFor([update(3)], first.phase).lines).toHaveLength(0);
  });
});
