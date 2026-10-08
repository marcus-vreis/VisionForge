import { describe, expect, it } from "vitest";

import { isCrossValidationReport } from "./report-shape";

describe("isCrossValidationReport", () => {
  const folds = [{ fold: 0, status: "success" }];

  it("recognises the classification K-fold report", () => {
    expect(
      isCrossValidationReport({
        fold_results: folds,
        mean_accuracy: 0.9,
        std_accuracy: 0.02,
        mean_f1: 0.88,
        std_f1: 0.03,
      }),
    ).toBe(true);
  });

  it("still recognises it with no spread: one finished fold has none", () => {
    expect(
      isCrossValidationReport({
        fold_results: folds,
        mean_accuracy: 0.9,
        std_accuracy: null,
        mean_f1: 0.88,
        std_f1: null,
      }),
    ).toBe(true);
  });

  it("still recognises it with no mean: stopped before any fold finished", () => {
    expect(
      isCrossValidationReport({
        fold_results: [{ fold: 0, status: "stopped" }],
        n_folds_ok: 0,
        mean_accuracy: null,
        std_accuracy: null,
        mean_f1: null,
        std_f1: null,
        stopped: true,
      }),
    ).toBe(true);
  });

  it("leaves the standalone tasks' K-fold report to its own view", () => {
    expect(
      isCrossValidationReport({
        fold_results: folds,
        aggregate: { miou: { mean: 0.5, std: null, n: 1 } },
        metric: "miou",
        n_folds: 3,
      }),
    ).toBe(false);
  });

  it("is not a K-fold report without its folds", () => {
    expect(isCrossValidationReport({ mean_accuracy: 0.9 })).toBe(false);
    expect(isCrossValidationReport({})).toBe(false);
  });
});
