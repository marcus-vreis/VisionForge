import { describe, expect, it } from "vitest";

import { curveSeries } from "./compare-curves";
import { upgradeAnomalyEpoch, upgradeRunHistory } from "./history-keys";

const OLD_EPOCH = { epoch: 1, train_loss: 0.1, val_auroc: 0.93, val_image_f1: 0.8 };

function run(history: Array<Record<string, unknown>>, extra: Record<string, unknown> = {}) {
  return { config: { task: "anomaly" }, history, ...extra };
}

describe("upgradeAnomalyEpoch", () => {
  it("reads the old AUROC as the test AUROC", () => {
    expect(upgradeAnomalyEpoch(OLD_EPOCH)).toEqual({
      epoch: 1,
      train_loss: 0.1,
      test_auroc: 0.93,
      val_image_f1: 0.8,
    });
  });

  it("leaves the record it was given as it was", () => {
    const record = { ...OLD_EPOCH };

    upgradeAnomalyEpoch(record);

    expect(record).toEqual(OLD_EPOCH);
  });

  it("keeps an undefined AUROC undefined", () => {
    expect(upgradeAnomalyEpoch({ epoch: 1, val_auroc: null })).toEqual({ epoch: 1, test_auroc: null });
  });

  it("keeps the new name's value when a record has both", () => {
    expect(upgradeAnomalyEpoch({ epoch: 1, val_auroc: 0.1, test_auroc: 0.9 })).toEqual({
      epoch: 1,
      test_auroc: 0.9,
    });
  });
});

describe("upgradeRunHistory", () => {
  it("renames the history of an anomaly run saved before the rename", () => {
    expect(upgradeRunHistory(run([OLD_EPOCH])).history).toEqual([upgradeAnomalyEpoch(OLD_EPOCH)]);
  });

  it("reads the task the server sent before the config's", () => {
    const upgraded = upgradeRunHistory(run([OLD_EPOCH], { task: "anomaly", config: {} }));

    expect(upgraded.history[0]).toHaveProperty("test_auroc", 0.93);
  });

  it("returns a current anomaly history with the same records", () => {
    const current = run([{ epoch: 1, test_auroc: 0.9 }]);

    expect(upgradeRunHistory(current).history).toEqual(current.history);
  });

  it("keeps a researcher's validation AUROC under its own name", () => {
    const custom = run([OLD_EPOCH], { task: "custom:defects" });

    expect(upgradeRunHistory(custom)).toBe(custom);
  });

  it("returns another task's run as it is", () => {
    const regression = { config: { task: "regression" }, history: [{ epoch: 1, val_r2: 0.5 }] };

    expect(upgradeRunHistory(regression)).toBe(regression);
  });

  it("lets an old anomaly run chart its AUROC first", () => {
    const series = curveSeries("anomaly", [upgradeRunHistory(run([OLD_EPOCH])).history]);

    expect(series.filter((s) => s.initial).map((s) => s.key)).toEqual(["test_auroc"]);
  });
});
