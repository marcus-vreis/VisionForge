import { describe, expect, it } from "vitest";
import {
  REFERENCE_SECTIONS,
  detailSections,
  headlineMetricKey,
  sectionStartsOpen,
} from "./run-detail-layout";

describe("detailSections", () => {
  it("puts a run's numbers and charts before its actions and its reference", () => {
    expect(detailSections({})).toEqual([
      "metrics",
      "charts",
      "actions",
      "tests",
      "dataset",
      "location",
      "environment",
      "training",
      "pipeline",
    ]);
  });

  it("names the group of a seed before its numbers", () => {
    expect(detailSections({ group_id: "g1" }).slice(0, 2)).toEqual(["member", "metrics"]);
  });

  it("leads a group with its own summary and offers no tests", () => {
    const sections = detailSections({ group: { kind: "replicates" } });
    expect(sections[0]).toBe("group");
    expect(sections).not.toContain("tests");
    expect(sections).not.toContain("member");
    expect(sections.indexOf("charts")).toBeLessThan(sections.indexOf("dataset"));
  });
});

describe("sectionStartsOpen", () => {
  it("folds the environment and the long reference blocks", () => {
    const open = REFERENCE_SECTIONS.filter(sectionStartsOpen);
    expect(open).toEqual(["dataset", "training"]);
    expect(sectionStartsOpen("environment")).toBe(false);
  });
});

describe("headlineMetricKey", () => {
  it("leads classification with the test accuracy", () => {
    expect(
      headlineMetricKey("classification", { best_val_loss: 0.2, test_accuracy: 0.91, test_f1: 0.9 }),
    ).toBe("test_accuracy");
  });

  it("falls back to the validation score of a run with no test split", () => {
    expect(headlineMetricKey("regression", { val_r2: 0.8, total_epochs: 3 })).toBe("val_r2");
    expect(headlineMetricKey("classification", { best_val_accuracy: 0.8 })).toBe(
      "best_val_accuracy",
    );
  });

  it("follows each task's History headline", () => {
    expect(headlineMetricKey("detection", { map50_95: 0.4, map50: 0.6 })).toBe("map50");
    expect(headlineMetricKey("segmentation", { test_dice: 0.7, test_miou: 0.6 })).toBe("test_miou");
    expect(headlineMetricKey("anomaly", { test_image_f1: 0.7, test_auroc: 0.95 })).toBe("test_auroc");
  });

  it("reads an unknown task as classification", () => {
    expect(headlineMetricKey("multiclass", { test_accuracy: 0.5 })).toBe("test_accuracy");
  });

  it("leads a researcher's task with its first reported number", () => {
    expect(
      headlineMetricKey("custom:leaf", { best_epoch: 2, total_epochs: 3, count_mae: 1.5, iou: 0.5 }),
    ).toBe("count_mae");
  });

  it("emphasises nothing that was never measured", () => {
    expect(headlineMetricKey("classification", { test_accuracy: null, total_epochs: 0 })).toBeNull();
    expect(headlineMetricKey("regression", { test_r2: Number.NaN })).toBeNull();
    expect(headlineMetricKey("custom:leaf", { total_epochs: 1 })).toBeNull();
  });
});
