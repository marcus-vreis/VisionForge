import { describe, expect, it } from "vitest";

import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { curveSeries, selectedCurves, toggleCurve } from "./compare-curves";

type Record1 = Record<string, unknown>;

/** A history of `n` epochs whose records carry `keys` (a number, or null where given). */
function history(keys: string[], n = 3, nulls: string[] = []): Record1[] {
  return Array.from({ length: n }, (_, i) => ({
    epoch: i + 1,
    ...Object.fromEntries(keys.map((k) => [k, nulls.includes(k) ? null : 0.1 * (i + 1)])),
  }));
}

const keysOf = (task: string, ...histories: Record1[][]) =>
  curveSeries(task, histories).map((s) => s.key);
const initialOf = (task: string, ...histories: Record1[][]) =>
  curveSeries(task, histories)
    .filter((s) => s.initial)
    .map((s) => s.key);

describe("curveSeries", () => {
  it("draws classification's val loss and val accuracy first, as it always did", () => {
    const h = history(["train_loss", "train_accuracy", "val_loss", "val_accuracy"]);
    expect(initialOf("classification", h)).toEqual(["val_loss", "val_accuracy"]);
    expect(keysOf("classification", h)).toEqual([
      "val_loss",
      "val_accuracy",
      "train_loss",
      "train_accuracy",
    ]);
  });

  it("gives detection its mAP and validation box loss, which the old chart never found", () => {
    const yolo = history([
      "map50",
      "map50_95",
      "precision",
      "recall",
      "box_loss",
      "train_box_loss",
      "train_cls_loss",
      "train_dfl_loss",
      "val_box_loss",
      "val_cls_loss",
      "val_dfl_loss",
    ]);
    expect(initialOf("detection", yolo)).toEqual(["map50_95", "val_box_loss"]);
    expect(keysOf("detection", yolo)).toEqual([
      "map50_95",
      "val_box_loss",
      "map50",
      "precision",
      "recall",
      "train_box_loss",
      "train_cls_loss",
      "train_dfl_loss",
      "val_cls_loss",
      "val_dfl_loss",
    ]);
  });

  it("never lists detection's box_loss beside val_box_loss, which is the same number", () => {
    const h = history(["map50", "box_loss", "val_box_loss"]);
    expect(keysOf("detection", h)).not.toContain("box_loss");
  });

  it("falls back to box_loss on a detection history that predates val_box_loss", () => {
    const old = history(["map50", "map50_95", "box_loss"]);
    expect(initialOf("detection", old)).toEqual(["map50_95", "box_loss"]);
  });

  it("falls back to mAP@50 where the backend measures no mAP@50-95, without listing it twice", () => {
    // torchvision: map50_95, precision and recall are null in every epoch.
    const tv = history(["map50", "map50_95", "precision", "recall", "box_loss", "val_box_loss"], 4, [
      "map50_95",
      "precision",
      "recall",
    ]);
    expect(initialOf("detection", tv)).toEqual(["map50", "val_box_loss"]);
    expect(keysOf("detection", tv)).toEqual(["map50", "val_box_loss"]);
  });

  it("gives regression its val loss and val R², and the other errors as options", () => {
    const h = history(["train_loss", "val_loss", "val_mse", "val_rmse", "val_mae", "val_r2"]);
    expect(initialOf("regression", h)).toEqual(["val_loss", "val_r2"]);
    expect(keysOf("regression", h)).toEqual([
      "val_loss",
      "val_r2",
      "val_rmse",
      "val_mae",
      "val_mse",
      "train_loss",
    ]);
  });

  it("gives segmentation its val loss and val mIoU, with Dice and pixel accuracy as options", () => {
    const h = history(["train_loss", "val_loss", "val_miou", "val_dice", "val_pixel_acc"]);
    expect(initialOf("segmentation", h)).toEqual(["val_loss", "val_miou"]);
    expect(keysOf("segmentation", h)).toEqual([
      "val_loss",
      "val_miou",
      "val_dice",
      "val_pixel_acc",
      "train_loss",
    ]);
  });

  it("gives anomaly its val AUROC, which is the curve that says something", () => {
    const h = history(["train_loss", "val_auroc", "val_image_f1", "val_threshold"]);
    expect(initialOf("anomaly", h)).toEqual(["val_auroc"]);
    expect(keysOf("anomaly", h)).toEqual(["val_auroc", "val_image_f1", "train_loss", "val_threshold"]);
  });

  it("offers only what some run measured", () => {
    expect(keysOf("regression", history(["val_loss", "val_r2"]))).toEqual(["val_loss", "val_r2"]);
    expect(keysOf("regression", history(["val_loss"], 3, ["val_loss"]))).toEqual([]);
    expect(keysOf("regression", [{ epoch: 1, val_loss: Number.NaN, val_r2: Infinity }])).toEqual([]);
  });

  it("offers a series that only one of the runs measured", () => {
    const a = history(["val_loss"]);
    const b = history(["val_loss", "val_r2"]);
    expect(keysOf("regression", a, b)).toEqual(["val_loss", "val_r2"]);
  });

  it("has no curve for a replicate group, whose history is empty, and draws the rest", () => {
    expect(keysOf("regression", [])).toEqual([]);
    expect(keysOf("regression", [], history(["val_loss", "val_r2"]))).toEqual(["val_loss", "val_r2"]);
  });

  it("reads a classification problem type, or a task it does not know, as classification", () => {
    const h = history(["val_loss", "val_accuracy"]);
    for (const task of ["binary", "multiclass", "something-new"]) {
      expect(initialOf(task, h), task).toEqual(["val_loss", "val_accuracy"]);
    }
  });

  it("offers a researcher's own task every numeric key under its own name", () => {
    const h = history(["train_loss", "val_iou", "val_f1", "note"]).map((r) => ({ ...r, note: "x" }));
    const series = curveSeries("custom:shapes", [h]);
    expect(series.map((s) => s.key)).toEqual(["train_loss", "val_iou", "val_f1"]);
    expect(series.map((s) => s.label)).toEqual([null, null, null]);
    expect(series.filter((s) => s.initial).map((s) => s.key)).toEqual(["train_loss", "val_iou"]);
  });

  it("reads which way each series improves from its name", () => {
    const all = history([
      "val_loss",
      "val_accuracy",
      "val_r2",
      "val_rmse",
      "val_mae",
      "val_miou",
      "val_auroc",
      "val_image_f1",
      "val_pixel_acc",
      "val_threshold",
    ]);
    const direction = Object.fromEntries(
      [...curveSeries("regression", [all]), ...curveSeries("anomaly", [all])].map((s) => [
        s.key,
        s.direction,
      ]),
    );
    expect(direction).toMatchObject({
      val_loss: "lower",
      val_r2: "higher",
      val_rmse: "lower",
      val_mae: "lower",
      val_auroc: "higher",
      val_image_f1: "higher",
      val_threshold: null,
    });
  });

  it("believes the directions the server sent over the name, and the name where it sent none", () => {
    const h = history(["train_loss", "val_score", "val_threshold", "val_iou"]);
    const series = curveSeries("custom:shapes", [h], {
      val_score: "lower",
      val_threshold: "higher",
    });
    const direction = Object.fromEntries(series.map((s) => [s.key, s.direction]));
    expect(direction).toEqual({
      train_loss: "lower",
      val_score: "lower",
      // A decision point per epoch is neither: the server's word does not change that.
      val_threshold: null,
      val_iou: "higher",
    });
  });

  it("reads the name alone when the server sent nothing", () => {
    const h = history(["val_score", "val_rmse"]);
    const direction = Object.fromEntries(
      curveSeries("custom:shapes", [h], undefined).map((s) => [s.key, s.direction]),
    );
    expect(direction).toEqual({ val_score: "higher", val_rmse: "lower" });
  });

  it("labels every built-in series with a name both dictionaries have", () => {
    const every = history(Object.keys(pt.compareRuns.curves));
    for (const task of ["classification", "detection", "regression", "segmentation", "anomaly"]) {
      for (const s of curveSeries(task, [every])) {
        expect(s.label, `${task}/${s.key}`).not.toBeNull();
        expect(s.label! in pt.compareRuns.curves, `pt ${s.key}`).toBe(true);
        expect(s.label! in en.compareRuns.curves, `en ${s.key}`).toBe(true);
      }
    }
  });
});

describe("selectedCurves", () => {
  const series = curveSeries("regression", [history(["val_loss", "val_r2", "val_rmse", "train_loss"])]);

  it("draws the initial series until something is picked", () => {
    expect(selectedCurves(series, null).map((s) => s.key)).toEqual(["val_loss", "val_r2"]);
  });

  it("draws what was picked, in the order the series are offered", () => {
    expect(selectedCurves(series, ["train_loss", "val_r2"]).map((s) => s.key)).toEqual([
      "val_r2",
      "train_loss",
    ]);
  });

  it("goes back to the initial series when the picks are not among these runs' any more", () => {
    expect(selectedCurves(series, ["val_miou"]).map((s) => s.key)).toEqual(["val_loss", "val_r2"]);
  });

  it("draws nothing when there is nothing to draw", () => {
    expect(selectedCurves([], null)).toEqual([]);
    expect(selectedCurves([], ["val_loss"])).toEqual([]);
  });
});

describe("toggleCurve", () => {
  it("adds a series that is not drawn", () => {
    expect(toggleCurve(["val_loss"], "val_r2")).toEqual(["val_loss", "val_r2"]);
  });

  it("removes a series that is drawn", () => {
    expect(toggleCurve(["val_loss", "val_r2"], "val_loss")).toEqual(["val_r2"]);
  });

  it("keeps the last series: a chart picker that empties the page has nothing left to click", () => {
    expect(toggleCurve(["val_r2"], "val_r2")).toEqual(["val_r2"]);
  });
});
