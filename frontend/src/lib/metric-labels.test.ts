import { describe, expect, it } from "vitest";

import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { metricLabel } from "./metric-labels";

/** The names each task writes into run.json `metrics`, read off real runs of
 *  the sample datasets (frontend/e2e/tasks.e2e.ts trains them). */
const RUN_METRICS: Record<string, string[]> = {
  classification: [
    "test_accuracy",
    "best_val_loss",
    "best_epoch",
    "total_epochs",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_auc_roc",
  ],
  regression: [
    "test_r2",
    "best_val_loss",
    "best_epoch",
    "total_epochs",
    "mse",
    "rmse",
    "mae",
    "r2",
    "test_mse",
    "test_rmse",
    "test_mae",
  ],
  segmentation: [
    "test_miou",
    "best_val_miou",
    "best_epoch",
    "total_epochs",
    "val_miou",
    "val_dice",
    "val_pixel_acc",
    "test_dice",
    "test_pixel_acc",
  ],
  anomaly: [
    "test_auroc",
    "best_auroc",
    "best_epoch",
    "total_epochs",
    "auroc",
    "threshold",
    "image_f1",
    "test_threshold",
    "test_image_f1",
  ],
  detection: ["map50", "map50_95", "precision", "recall", "box_loss", "best_epoch", "total_epochs"],
};

describe("metricLabel", () => {
  it("names every metric the five tasks report, in both languages", () => {
    for (const dict of [pt, en]) {
      for (const [task, keys] of Object.entries(RUN_METRICS)) {
        for (const key of keys) {
          const label = metricLabel(dict, key);
          expect(label, `${task}: ${key}`).not.toBe(key);
          expect(label, `${task}: ${key}`).not.toMatch(/_/);
        }
      }
    }
  });

  it("keeps the labels the run detail already had", () => {
    expect(metricLabel(pt, "test_accuracy")).toBe("Acurácia (teste)");
    expect(metricLabel(pt, "test_f1")).toBe("F1 (teste)");
    expect(metricLabel(pt, "precision")).toBe("Precisão");
    expect(metricLabel(pt, "map50")).toBe("mAP@50");
    expect(metricLabel(pt, "box_loss")).toBe("Box loss (val)");
    expect(metricLabel(pt, "total_epochs")).toBe("Épocas treinadas");
    expect(metricLabel(en, "test_accuracy")).toBe("Accuracy (test)");
  });

  it("says which split a regression or segmentation number comes from", () => {
    expect(metricLabel(pt, "r2")).toBe("R² (val)");
    expect(metricLabel(pt, "test_r2")).toBe("R² (teste)");
    expect(metricLabel(pt, "mse")).toBe("MSE (val)");
    expect(metricLabel(pt, "test_mse")).toBe("MSE (teste)");
    expect(metricLabel(pt, "val_miou")).toBe("mIoU (val)");
    expect(metricLabel(pt, "best_val_miou")).toBe("Melhor mIoU (val)");
    expect(metricLabel(en, "test_miou")).toBe("mIoU (test)");
  });

  it("never calls an anomaly score a validation one: the task has no validation split", () => {
    // core/anomaly_trainer.py scores every epoch on the labelled test split.
    for (const dict of [pt, en]) {
      for (const key of ["auroc", "threshold", "image_f1", "best_auroc"]) {
        expect(metricLabel(dict, key), key).not.toMatch(/\bval\b/i);
      }
    }
    expect(metricLabel(pt, "auroc")).toBe("AUROC (melhor época)");
    expect(metricLabel(pt, "test_auroc")).toBe("AUROC (teste)");
  });

  it("shows a researcher's own metric by its own name", () => {
    expect(metricLabel(pt, "count_error")).toBe("count_error");
  });
});
