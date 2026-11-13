import { describe, expect, it } from "vitest";
import { folderName, historyTargetForResult, resolveHistoryTarget } from "./history-target";

const JOB = "exp_20261112_135436_814235";
const result = (report: Record<string, unknown>) => ({ run_id: JOB, report });

describe("folderName", () => {
  it("takes the last segment of a Windows or POSIX path", () => {
    expect(folderName("C:\\out\\models\\exp\\20261112_135436_814235")).toBe(
      "20261112_135436_814235",
    );
    expect(folderName("/content/out/models/exp/20261112_135436/")).toBe("20261112_135436");
  });

  it("is null for anything that names no folder", () => {
    expect(folderName("")).toBeNull();
    expect(folderName(null)).toBeNull();
    expect(folderName(3)).toBeNull();
  });
});

describe("historyTargetForResult", () => {
  it("names a classification or transfer-learning run by its train folder", () => {
    expect(
      historyTargetForResult(
        result({ train: { run_dir: "C:\\m\\exp\\20261112_135436_814235" }, eval: {} }),
      ),
    ).toEqual({ runId: "20261112_135436_814235" });
  });

  it("names regression, segmentation and anomaly runs the same way", () => {
    expect(historyTargetForResult(result({ train: { run_dir: "/m/reg/20261112_1" } }))).toEqual({
      runId: "20261112_1",
    });
  });

  it("names a detection run by its detection folder", () => {
    expect(
      historyTargetForResult(result({ detection: { run_dir: "/m/det/20261112_2", best_map50: 0.5 } })),
    ).toEqual({ runId: "20261112_2" });
  });

  it("names a researcher's task run by the folder at the top of its report", () => {
    expect(
      historyTargetForResult(result({ task: "custom:leaf", run_dir: "/m/leaf/20261112_3" })),
    ).toEqual({ runId: "20261112_3" });
  });

  it("names a strategies job (replicates, comparison, task K-fold, sweep) by the job id", () => {
    const replicates = { seeds: [1, 2], aggregates: {}, report_dir: "/r/x" };
    const comparison = { kind: "replicated_comparison", variants: {}, report_dir: "/r/y" };
    const kfold = { fold_results: [{ run_dir: "/m/a/f0" }], aggregate: {}, report_dir: "/r/z" };
    const sweep = { mode: "grid", trials: [], best_trial: { run_dir: "/m/a/t0" }, report_dir: "/r/s" };
    const models = { metric: "r2", trials: [], report_dir: "/r/c" };
    for (const report of [replicates, comparison, kfold, sweep, models]) {
      expect(historyTargetForResult(result(report))).toEqual({ runId: JOB });
    }
  });

  it("finds a classification K-fold through its folds", () => {
    expect(
      historyTargetForResult(
        result({
          fold_results: [{ run_dir: "/m/a/f0" }, { run_dir: null }, { run_dir: "/m/a/f2" }],
          mean_accuracy: 0.9,
        }),
      ),
    ).toEqual({ childRunIds: ["f0", "f2"] });
  });

  it("finds a classification grid search through its best trial", () => {
    expect(
      historyTargetForResult(
        result({ best_trial: { run_dir: "/m/a/t3" }, total_trials: 4, successful_trials: 4 }),
      ),
    ).toEqual({ childRunIds: ["t3"] });
  });

  it("finds a classification model comparison through its ranked runs", () => {
    expect(
      historyTargetForResult(
        result({ top_3: [{ run_dir: "/m/a/r1" }, { run_dir: "/m/b/r2" }], total_ran: 2, failed_count: 0 }),
      ),
    ).toEqual({ childRunIds: ["r1", "r2"] });
  });

  it("names nothing when the report names no folder", () => {
    expect(historyTargetForResult(result({}))).toBeNull();
    expect(historyTargetForResult(result({ train: { run_dir: null } }))).toBeNull();
    expect(historyTargetForResult(result({ best_trial: null, total_trials: 0 }))).toBeNull();
    expect(historyTargetForResult(result({ fold_results: [], mean_accuracy: null }))).toBeNull();
  });

  it("does not take a custom-looking run_dir from a report that is not a custom task's", () => {
    expect(historyTargetForResult(result({ run_dir: "/m/x/20261112_9" }))).toBeNull();
  });
});

describe("resolveHistoryTarget", () => {
  const runs = [
    { run_id: "single", group_id: null },
    { run_id: "group_a", group_id: null },
    { run_id: "f0", group_id: "group_a" },
    { run_id: "orphan", group_id: null },
    { run_id: "lost_child", group_id: "group_never_written" },
  ];

  it("opens a listed run by its id", () => {
    expect(resolveHistoryTarget({ runId: "single" }, runs)).toBe("single");
  });

  it("names nothing while the run is not listed yet", () => {
    expect(resolveHistoryTarget({ runId: "missing" }, runs)).toBeNull();
  });

  it("opens the group of the first listed unit", () => {
    expect(resolveHistoryTarget({ childRunIds: ["not_yet", "f0"] }, runs)).toBe("group_a");
  });

  it("never opens a unit that has no listed group", () => {
    expect(resolveHistoryTarget({ childRunIds: ["orphan"] }, runs)).toBeNull();
    expect(resolveHistoryTarget({ childRunIds: ["lost_child"] }, runs)).toBeNull();
  });
});
