import { describe, expect, it } from "vitest";

import { curvePoints, curveQuantity, runCurveGroups } from "./run-curves";

const classification = [
  { epoch: 1, train_loss: 0.9, val_loss: 0.8, train_accuracy: 0.5, val_accuracy: 0.6 },
  { epoch: 2, train_loss: 0.4, val_loss: 0.5, train_accuracy: 0.8, val_accuracy: 0.7 },
];

describe("curveQuantity", () => {
  it("drops the split prefix and nothing else", () => {
    expect(curveQuantity("train_loss")).toBe("loss");
    expect(curveQuantity("val_box_loss")).toBe("box_loss");
    expect(curveQuantity("box_loss")).toBe("box_loss");
    expect(curveQuantity("map50_95")).toBe("map50_95");
  });
});

describe("runCurveGroups", () => {
  it("pairs train and validation of the same quantity, validation first", () => {
    const groups = runCurveGroups("classification", classification);
    expect(groups.map((g) => g.id)).toEqual(["loss", "accuracy"]);
    expect(groups[0].series.map((s) => [s.key, s.role])).toEqual([
      ["val_loss", "val"],
      ["train_loss", "train"],
    ]);
    expect(groups[1].series.map((s) => s.key)).toEqual(["val_accuracy", "train_accuracy"]);
  });

  it("charts a validation series alone when the run kept no train number for it", () => {
    const history = [{ epoch: 1, val_loss: 0.4, val_miou: 0.5 }];
    const groups = runCurveGroups("segmentation", history);
    expect(groups.map((g) => g.series.map((s) => s.key))).toEqual([["val_loss"], ["val_miou"]]);
  });

  it("gives only the task's leading quantities a chart", () => {
    const history = [
      {
        epoch: 1,
        map50_95: 0.3,
        map50: 0.5,
        precision: 0.6,
        recall: 0.4,
        val_box_loss: 1.2,
        train_box_loss: 1.4,
        train_cls_loss: 2.0,
      },
    ];
    const groups = runCurveGroups("detection", history);
    expect(groups.map((g) => g.id)).toEqual(["map50_95", "box_loss"]);
    expect(groups[1].series.map((s) => s.key)).toEqual(["val_box_loss", "train_box_loss"]);
  });

  it("does not draw detection's older box_loss beside val_box_loss", () => {
    const history = [{ epoch: 1, map50: 0.5, box_loss: 1.2, train_box_loss: 1.4 }];
    const groups = runCurveGroups("detection", history);
    expect(groups.find((g) => g.id === "box_loss")?.series.map((s) => s.key)).toEqual([
      "box_loss",
      "train_box_loss",
    ]);
  });

  it("is empty for a run with no history", () => {
    expect(runCurveGroups("classification", [])).toEqual([]);
  });

  it("is empty when nothing the task charts was measured", () => {
    expect(runCurveGroups("classification", [{ epoch: 1, lr: 0.001 }])).toEqual([]);
  });
});

describe("curvePoints", () => {
  it("keeps the epoch and turns a missing or non-finite value into a gap", () => {
    const history = [
      { epoch: 1, val_loss: 0.5 },
      { epoch: 2, val_loss: null },
      { epoch: 3 },
      { epoch: 4, val_loss: Number.NaN },
      { epoch: 5, val_loss: 0.1 },
    ];
    expect(curvePoints(history, "val_loss")).toEqual([
      { x: 1, y: 0.5 },
      { x: 2, y: null },
      { x: 3, y: null },
      { x: 4, y: null },
      { x: 5, y: 0.1 },
    ]);
  });

  it("skips a record with no epoch", () => {
    expect(curvePoints([{ val_loss: 0.5 }, { epoch: 2, val_loss: 0.4 }], "val_loss")).toEqual([
      { x: 2, y: 0.4 },
    ]);
  });
});
