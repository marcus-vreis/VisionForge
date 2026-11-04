import { describe, expect, it } from "vitest";

import type { TrainingEvent } from "../types/run";
import {
  bestEpoch,
  bestRule,
  curveOf,
  formatValue,
  liveUnits,
  metricKey,
  overridesSummary,
  type EpochEnd,
} from "./live-results";
import { storySpecFor } from "./story-family";

function epoch(n: number, fields: Partial<EpochEnd> = {}): EpochEnd {
  return {
    event: "epoch_end",
    epoch: n,
    total_epochs: 5,
    train_loss: 1 / n,
    val_accuracy: 0,
    ...fields,
  } as EpochEnd;
}

describe("liveUnits", () => {
  it("reads a run without trials as one unit", () => {
    const events: TrainingEvent[] = [
      { event: "start", total_epochs: 2 },
      epoch(1),
      epoch(2),
      { event: "end", total_epochs: 2 },
    ];
    const units = liveUnits(events);
    expect(units).toHaveLength(1);
    expect(units[0]).toMatchObject({ index: null, overrides: null, closed: false });
    expect(units[0].epochs.map((e) => e.epoch)).toEqual([1, 2]);
  });

  it("splits a sweep by trial and marks the closed ones", () => {
    const events: TrainingEvent[] = [
      { event: "trial_start", trial_index: 0, total_trials: 2, overrides: { lr: 0.1 }, seed: 1 },
      epoch(1),
      { event: "trial_end", trial_index: 0, total_trials: 2, total_epochs: 1 },
      { event: "trial_start", trial_index: 1, total_trials: 2, overrides: { lr: 0.01 }, seed: 1 },
      epoch(1),
      epoch(2),
    ];
    const units = liveUnits(events);
    expect(units.map((u) => [u.index, u.closed, u.epochs.length])).toEqual([
      [0, true, 1],
      [1, false, 2],
    ]);
    expect(units[1].overrides).toEqual({ lr: 0.01 });
  });
});

describe("bestRule", () => {
  it("follows the checkpoint rule of each built-in trainer", () => {
    const spec = (task: string) => storySpecFor(task, { model: { name: "resnet18" } });
    expect(bestRule(spec("classification"), [])).toEqual({ key: "val_loss", direction: "lower" });
    expect(bestRule(spec("regression"), [])).toEqual({ key: "val_loss", direction: "lower" });
    expect(bestRule(spec("segmentation"), [])).toEqual({ key: "val_miou", direction: "higher" });
  });

  it("uses mAP@50-95 when the detector measures it, else mAP@50", () => {
    const spec = storySpecFor("detection", { model: { name: "yolov8n" } });
    expect(bestRule(spec, [epoch(1, { map50: 0.4, map50_95: 0.2 })])?.key).toBe("map50_95");
    // torchvision reports a null mAP@50-95.
    expect(bestRule(spec, [epoch(1, { map50: 0.4, map50_95: null })])?.key).toBe("map50");
  });

  it("keeps the autoencoder's lowest training loss and has no rule for PatchCore", () => {
    const ae = storySpecFor("anomaly", { model: { name: "autoencoder" } });
    const pc = storySpecFor("anomaly", { model: { name: "patchcore" } });
    expect(ae.family).toBe("autoencoder");
    expect(pc.family).toBe("patchcore");
    expect(bestRule(ae, [])).toEqual({ key: "train_loss", direction: "lower" });
    expect(bestRule(pc, [])).toBeNull();
  });

  it("states no rule for a researcher's own task, whose direction the stream lacks", () => {
    expect(bestRule(storySpecFor("custom:my_task", {}), [])).toBeNull();
  });
});

describe("bestEpoch", () => {
  it("picks the lowest value for a lower-is-better rule", () => {
    const epochs = [
      epoch(1, { val_loss: 0.9 }),
      epoch(2, { val_loss: 0.4 }),
      epoch(3, { val_loss: 0.6 }),
    ];
    expect(bestEpoch(epochs, { key: "val_loss", direction: "lower" })).toEqual({
      epoch: 2,
      key: "val_loss",
      value: 0.4,
      direction: "lower",
    });
  });

  it("keeps the earlier epoch on a tie, as the trainers only save on a strict gain", () => {
    const epochs = [epoch(1, { map50: 0.5 }), epoch(2, { map50: 0.7 }), epoch(3, { map50: 0.7 })];
    expect(bestEpoch(epochs, { key: "map50", direction: "higher" })?.epoch).toBe(2);
  });

  it("skips epochs that did not report the number", () => {
    const epochs = [epoch(1, { map50_95: null }), epoch(2, { map50_95: 0.3 })];
    expect(bestEpoch(epochs, { key: "map50_95", direction: "higher" })?.epoch).toBe(2);
  });

  it("is null without a rule or without any value", () => {
    expect(bestEpoch([epoch(1)], null)).toBeNull();
    expect(bestEpoch([epoch(1)], { key: "val_miou", direction: "higher" })).toBeNull();
  });
});

describe("curveOf and metricKey", () => {
  it("keeps only the finite values of a field", () => {
    const epochs = [
      epoch(1, { val_loss: 0.8 }),
      epoch(2, { val_loss: Number.NaN }),
      epoch(3, { val_loss: 0.5 }),
    ];
    expect(curveOf(epochs, "val_loss")).toEqual([
      { epoch: 1, value: 0.8 },
      { epoch: 3, value: 0.5 },
    ]);
  });

  it("names the field of each headline metric", () => {
    expect(metricKey("accuracy")).toBe("val_accuracy");
    expect(metricKey("map50")).toBe("map50");
    expect(metricKey("r2")).toBe("val_r2");
    expect(metricKey("miou")).toBe("val_miou");
    expect(metricKey("auroc")).toBe("val_auroc");
    expect(metricKey("primary")).toBe("val_accuracy");
  });
});

describe("overridesSummary", () => {
  it("writes name=value pairs, objects as JSON", () => {
    expect(overridesSummary({ lr: 0.01, model: { name: "resnet18" } })).toBe(
      'lr=0.01, model={"name":"resnet18"}',
    );
    expect(overridesSummary(null)).toBe("");
  });
});

describe("formatValue", () => {
  it("writes a loss trained near zero so it does not read as zero", () => {
    expect(formatValue(0.00023174)).toBe("2.32e-4");
    expect(formatValue(0)).toBe("0");
    expect(formatValue(0.3)).toBe("0.3000");
    expect(formatValue(123.456)).toBe("123.5");
  });
});
