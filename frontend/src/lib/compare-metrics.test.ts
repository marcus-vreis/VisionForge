import { describe, expect, it } from "vitest";

import {
  cardMetrics,
  distinctTasks,
  extremeIndexes,
  inferMetricDirection,
  isCustomTaskKey,
  metricRows,
  numericMetric,
  runTaskKey,
} from "./compare-metrics";

const keys = (rows: ReturnType<typeof metricRows>) => rows.map((r) => r.key);
const directions = (rows: ReturnType<typeof metricRows>) =>
  Object.fromEntries(rows.map((r) => [r.key, r.direction]));

describe("inferMetricDirection", () => {
  it("reads error metrics as lower-is-better, as the backend ranks them", () => {
    for (const name of ["loss", "best_val_loss", "mae", "test_rmse", "mse", "box_loss", "error_rate"]) {
      expect(inferMetricDirection(name), name).toBe("lower");
    }
  });

  it("reads everything else as higher-is-better", () => {
    for (const name of ["accuracy", "f1", "auc_roc", "r2", "miou", "dice", "map50_95", "auroc"]) {
      expect(inferMetricDirection(name), name).toBe("higher");
    }
  });
});

describe("numericMetric", () => {
  it("keeps finite numbers, zero included", () => {
    expect(numericMetric(0)).toBe(0);
    expect(numericMetric(0.82)).toBe(0.82);
  });

  it("drops what is not a measurement", () => {
    for (const value of [null, undefined, "0.8", Number.NaN, Infinity, -Infinity, {}]) {
      expect(numericMetric(value)).toBeNull();
    }
  });
});

describe("runTaskKey", () => {
  it("believes the server's task", () => {
    expect(runTaskKey({ task: "custom:shapes", config: {} })).toBe("custom:shapes");
    expect(runTaskKey({ task: "regression", config: { task: "binary" } })).toBe("regression");
  });

  it("falls back to config.task for the standalone tasks, as the backend does", () => {
    for (const task of ["detection", "regression", "segmentation", "anomaly"]) {
      expect(runTaskKey({ config: { task } }), task).toBe(task);
    }
  });

  it("reads a classification problem type as classification", () => {
    for (const task of ["binary", "multiclass", "multilabel", undefined]) {
      expect(runTaskKey({ config: { task } }), String(task)).toBe("classification");
    }
  });
});

describe("distinctTasks", () => {
  it("lists each task once, in the order the runs name them", () => {
    const runs = [
      { task: "detection", config: {} },
      { task: "classification", config: {} },
      { task: "detection", config: {} },
    ];
    expect(distinctTasks(runs)).toEqual(["detection", "classification"]);
  });

  it("is a single task when the runs agree, whatever the problem type", () => {
    const runs = [{ config: { task: "binary" } }, { config: { task: "multiclass" } }, { config: {} }];
    expect(distinctTasks(runs)).toEqual(["classification"]);
  });
});

describe("isCustomTaskKey", () => {
  it("tells a researcher's task from a built-in", () => {
    expect(isCustomTaskKey("custom:shapes")).toBe(true);
    expect(isCustomTaskKey("detection")).toBe(false);
  });
});

describe("metricRows, classification", () => {
  const run = {
    metrics: {
      best_val_loss: 0.31,
      best_epoch: 7,
      total_epochs: 10,
      test_accuracy: 0.9,
      test_f1: 0.88,
      test_precision: 0.87,
      test_recall: 0.89,
      test_auc_roc: 0.95,
    },
  };

  it("keeps the rows the table always had, in their order", () => {
    expect(keys(metricRows("classification", [run, run]))).toEqual([
      "best_val_loss",
      "best_epoch",
      "total_epochs",
      "test_accuracy",
      "test_f1",
      "test_precision",
      "test_recall",
      "test_auc_roc",
    ]);
  });

  it("leaves out a row no run measured", () => {
    const untested = { metrics: { best_val_loss: 0.4, best_epoch: 3, total_epochs: 5 } };
    expect(keys(metricRows("classification", [untested, untested]))).toEqual([
      "best_val_loss",
      "best_epoch",
      "total_epochs",
    ]);
  });

  it("keeps a row one run measured, so the other shows a dash", () => {
    const untested = { metrics: { best_val_loss: 0.4, best_epoch: 3, total_epochs: 5 } };
    expect(keys(metricRows("classification", [untested, run]))).toContain("test_accuracy");
  });

  it("drops a row whose only values are null, as in a run stopped before an epoch", () => {
    const stopped = { metrics: { best_val_loss: null, best_epoch: 0, total_epochs: 0 } };
    expect(keys(metricRows("classification", [stopped, stopped]))).toEqual(["best_epoch", "total_epochs"]);
  });

  it("marks the loss lower-is-better, the scores higher, and the epochs neither", () => {
    expect(directions(metricRows("classification", [run]))).toEqual({
      best_val_loss: "lower",
      best_epoch: null,
      total_epochs: null,
      test_accuracy: "higher",
      test_f1: "higher",
      test_precision: "higher",
      test_recall: "higher",
      test_auc_roc: "higher",
    });
  });

  it("lists a K-fold the same way: its means are written under the same names", () => {
    const kfold = {
      metrics: {
        total_epochs: 30,
        test_accuracy: 0.9,
        test_f1: 0.88,
        best_val_loss: 0.3,
        fold_results: [{ fold: 0 }],
        cv_aggregate: { n_folds: 3 },
      },
    };
    expect(keys(metricRows("classification", [kfold]))).toEqual([
      "best_val_loss",
      "total_epochs",
      "test_accuracy",
      "test_f1",
    ]);
  });
});

describe("metricRows, detection", () => {
  // What core/detection_trainer.py writes: validation scores at the best epoch.
  const run = {
    metrics: {
      map50_95: 0.52,
      map50: 0.78,
      precision: 0.8,
      recall: 0.7,
      box_loss: 0.9,
      best_epoch: 12,
      total_epochs: 20,
    },
  };

  it("lists mAP, precision and recall, not the classification rows", () => {
    expect(keys(metricRows("detection", [run, run]))).toEqual([
      "best_epoch",
      "total_epochs",
      "map50_95",
      "map50",
      "precision",
      "recall",
    ]);
  });

  it("reads all of them as higher-is-better", () => {
    const rows = directions(metricRows("detection", [run]));
    expect(rows.map50_95).toBe("higher");
    expect(rows.map50).toBe("higher");
    expect(rows.precision).toBe("higher");
    expect(rows.recall).toBe("higher");
  });

  it("shows nothing of a classification run's names even when the run has them", () => {
    const odd = { metrics: { test_accuracy: 0.9, map50: 0.5 } };
    expect(keys(metricRows("detection", [odd]))).toEqual(["map50"]);
  });
});

describe("metricRows, regression", () => {
  // core/regression_trainer.py (validation) plus blocks/regression.py (test split).
  const run = {
    metrics: {
      best_val_loss: 0.2,
      best_epoch: 4,
      total_epochs: 8,
      mse: 0.25,
      rmse: 0.5,
      mae: 0.4,
      r2: 0.8,
      test_mse: 0.3,
      test_rmse: 0.55,
      test_mae: 0.45,
      test_r2: 0.75,
    },
  };

  it("lists R², RMSE and MAE of the test split, then of validation", () => {
    expect(keys(metricRows("regression", [run, run]))).toEqual([
      "best_val_loss",
      "best_epoch",
      "total_epochs",
      "test_r2",
      "test_rmse",
      "test_mae",
      "r2",
      "rmse",
      "mae",
    ]);
  });

  it("reads R² as higher-is-better and the errors as lower-is-better", () => {
    const rows = directions(metricRows("regression", [run]));
    expect(rows.test_r2).toBe("higher");
    expect(rows.r2).toBe("higher");
    expect(rows.test_rmse).toBe("lower");
    expect(rows.rmse).toBe("lower");
    expect(rows.test_mae).toBe("lower");
    expect(rows.mae).toBe("lower");
    expect(rows.best_val_loss).toBe("lower");
  });

  it("drops the test rows for runs that never scored a test split", () => {
    const validationOnly = Object.fromEntries(
      Object.entries(run.metrics).filter(([name]) => !name.startsWith("test_")),
    );
    expect(keys(metricRows("regression", [{ metrics: validationOnly }]))).toEqual([
      "best_val_loss",
      "best_epoch",
      "total_epochs",
      "r2",
      "rmse",
      "mae",
    ]);
  });
});

describe("metricRows, segmentation", () => {
  const run = {
    metrics: {
      best_val_miou: 0.61,
      best_epoch: 9,
      total_epochs: 15,
      miou: 0.61,
      dice: 0.74,
      pixel_acc: 0.93,
      test_miou: 0.58,
      test_dice: 0.71,
      test_pixel_acc: 0.92,
    },
  };

  it("lists mIoU, Dice and pixel accuracy, test split first", () => {
    expect(keys(metricRows("segmentation", [run, run]))).toEqual([
      "best_epoch",
      "total_epochs",
      "test_miou",
      "test_dice",
      "test_pixel_acc",
      "miou",
      "dice",
      "pixel_acc",
    ]);
  });

  it("reads them all as higher-is-better", () => {
    const rows = directions(metricRows("segmentation", [run]));
    expect(rows.test_miou).toBe("higher");
    expect(rows.dice).toBe("higher");
    expect(rows.pixel_acc).toBe("higher");
  });
});

describe("metricRows, anomaly", () => {
  const run = {
    metrics: {
      best_auroc: 0.9,
      best_epoch: 3,
      total_epochs: 5,
      auroc: 0.9,
      threshold: 1.7,
      image_f1: 0.8,
      test_auroc: 0.88,
      test_threshold: 1.7,
      test_image_f1: 0.79,
    },
  };

  it("lists AUROC, image F1 and the threshold", () => {
    expect(keys(metricRows("anomaly", [run, run]))).toEqual([
      "best_epoch",
      "total_epochs",
      "test_auroc",
      "test_image_f1",
      "test_threshold",
      "auroc",
      "image_f1",
      "threshold",
    ]);
  });

  it("gives the threshold no direction: a decision point is not better when larger", () => {
    const rows = directions(metricRows("anomaly", [run]));
    expect(rows.test_threshold).toBeNull();
    expect(rows.threshold).toBeNull();
    expect(rows.test_auroc).toBe("higher");
    expect(rows.image_f1).toBe("higher");
  });
});

describe("metricRows, a researcher's own task", () => {
  const run = {
    metrics: {
      best_epoch: 2,
      total_epochs: 4,
      score: 0.7,
      edge_error: 0.12,
      note: "ok",
      missing: null,
      nested: { a: 1 },
    },
  };

  it("lists the bookkeeping and then every numeric metric, under its own name", () => {
    const rows = metricRows("custom:shapes", [run, run]);
    expect(keys(rows)).toEqual(["best_epoch", "total_epochs", "score", "edge_error"]);
  });

  it("names the bookkeeping from the dictionary and leaves the researcher's names alone", () => {
    const labels = Object.fromEntries(
      metricRows("custom:shapes", [run, run]).map((r) => [r.key, r.label]),
    );
    expect(labels).toEqual({
      best_epoch: "best_epoch",
      total_epochs: "total_epochs",
      score: null,
      edge_error: null,
    });
  });

  it("unites the names the runs reported, first seen first", () => {
    const other = { metrics: { total_epochs: 3, extra: 1 } };
    expect(keys(metricRows("custom:shapes", [run, other]))).toEqual([
      "best_epoch",
      "total_epochs",
      "score",
      "edge_error",
      "extra",
    ]);
  });

  it("falls back to the name for the direction", () => {
    const rows = directions(metricRows("custom:shapes", [run]));
    expect(rows.score).toBe("higher");
    expect(rows.edge_error).toBe("lower");
    expect(rows.best_epoch).toBeNull();
  });

  it("believes the direction the task declared over the name", () => {
    const rows = directions(metricRows("custom:shapes", [run], { score: "lower", edge_error: "higher" }));
    expect(rows.score).toBe("lower");
    expect(rows.edge_error).toBe("higher");
  });

  it("ignores a declaration that is not a direction", () => {
    const rows = directions(metricRows("custom:shapes", [run], { score: "sideways" }));
    expect(rows.score).toBe("higher");
  });

  it("never reads the declared directions for a built-in task", () => {
    const rows = directions(metricRows("detection", [{ metrics: { map50: 0.5 } }], { map50: "lower" }));
    expect(rows.map50).toBe("higher");
  });

  it("lists nothing of a task it has never heard of beyond what the runs reported", () => {
    expect(keys(metricRows("future_task", [{ metrics: { quality: 1 } }]))).toEqual(["quality"]);
  });
});

describe("extremeIndexes", () => {
  it("finds the highest value of a higher-is-better row", () => {
    expect(extremeIndexes([0.7, 0.9, 0.8], "higher")).toEqual([1]);
  });

  it("finds the lowest value of a lower-is-better row", () => {
    expect(extremeIndexes([0.7, 0.2, 0.8], "lower")).toEqual([1]);
  });

  it("marks every tied extreme", () => {
    expect(extremeIndexes([0.9, 0.5, 0.9], "higher")).toEqual([0, 2]);
  });

  it("marks nothing when the values are all equal", () => {
    expect(extremeIndexes([0.5, 0.5], "higher")).toEqual([]);
  });

  it("marks nothing when only one run measured it", () => {
    expect(extremeIndexes([0.9, null], "higher")).toEqual([]);
  });

  it("skips a run with no value and still finds the extreme of the rest", () => {
    expect(extremeIndexes([null, 0.4, 0.6], "lower")).toEqual([1]);
  });

  it("marks nothing in a row with no direction", () => {
    expect(extremeIndexes([3, 9], null)).toEqual([]);
  });

  it("treats zero as a value, not as a missing one", () => {
    expect(extremeIndexes([0, 0.4], "lower")).toEqual([0]);
  });
});

describe("cardMetrics", () => {
  // The names the server projects per task (`_SUMMARY_METRIC_KEYS` in
  // gui/api/routes.py), as a run's `final_metrics` carries them.
  const PROJECTED: Record<string, string[]> = {
    classification: ["accuracy", "f1", "val_loss"],
    detection: ["map50", "map50_95"],
    regression: ["r2", "mae", "rmse"],
    segmentation: ["miou", "dice", "pixel_acc"],
    anomaly: ["auroc", "f1"],
  };
  const summary = (names: string[]) => Object.fromEntries(names.map((n, i) => [n, 0.5 + i / 10]));
  const labelsOf = (task: string, names: string[]) =>
    cardMetrics(task, summary(names)).map((m) => m.label);

  it("names every number the server projects for a built-in task from the table", () => {
    for (const [task, names] of Object.entries(PROJECTED)) {
      const got = cardMetrics(task, summary(names));
      expect(
        got.map((m) => m.key),
        task,
      ).toEqual(names);
      for (const m of got) expect(m.label, `${task}/${m.key}`).not.toBeNull();
    }
  });

  it("reads the held-out split's row where the server reads it", () => {
    expect(labelsOf("regression", ["r2", "mae", "rmse"])).toEqual([
      "test_r2",
      "test_mae",
      "test_rmse",
    ]);
    expect(labelsOf("segmentation", ["miou", "dice", "pixel_acc"])).toEqual([
      "test_miou",
      "test_dice",
      "test_pixel_acc",
    ]);
    expect(labelsOf("anomaly", ["auroc", "f1"])).toEqual(["test_auroc", "test_image_f1"]);
    expect(labelsOf("classification", ["accuracy", "f1", "val_loss"])).toEqual([
      "test_accuracy",
      "test_f1",
      "best_val_loss",
    ]);
  });

  it("keeps detection on its validation rows, the only ones it has", () => {
    expect(labelsOf("detection", ["map50", "map50_95"])).toEqual(["map50", "map50_95"]);
  });

  it("keeps the order the server sent and shows at most three", () => {
    const got = cardMetrics("regression", { rmse: 3, r2: 0.9, mae: 2, test_mse: 9 });
    expect(got.map((m) => m.key)).toEqual(["rmse", "r2", "mae"]);
  });

  it("leaves out a metric that was never measured but keeps a zero", () => {
    const got = cardMetrics("anomaly", { auroc: Number.NaN, f1: 0 });
    expect(got.map((m) => m.key)).toEqual(["f1"]);
    expect(cardMetrics("regression", { r2: Infinity, mae: null, rmse: "3" })).toEqual([]);
  });

  it("reads a classification problem type, or a task it does not know, as classification", () => {
    for (const task of ["binary", "multiclass", "multilabel", "something-new"]) {
      expect(cardMetrics(task, { accuracy: 0.9 })[0].label, task).toBe("test_accuracy");
    }
  });

  it("keeps a researcher's own task under the names it reported", () => {
    expect(cardMetrics("custom:shapes", { iou: 0.7, val_loss: 0.2, f1: 0.5, extra: 1 })).toEqual([
      { key: "iou", label: null },
      { key: "val_loss", label: null },
      { key: "f1", label: null },
    ]);
  });

  it("shows a name the table does not know under its own name instead of dropping it", () => {
    expect(cardMetrics("regression", { r2: 0.9, nse: 0.8 })).toEqual([
      { key: "r2", label: "test_r2" },
      { key: "nse", label: null },
    ]);
  });

  it("never labels a card with the training bookkeeping", () => {
    expect(labelsOf("detection", ["best_epoch", "total_epochs"])).toEqual([null, null]);
  });
});
