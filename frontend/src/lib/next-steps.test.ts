import { describe, expect, it } from "vitest";
import { nextSteps, type NextStepsInput } from "./next-steps";

/** A finished, tested run that plateaued and belongs to a replicate group:
 *  no rule holds for it. Each case changes only what its rule reads. */
function run(overrides: Partial<NextStepsInput> = {}): NextStepsInput {
  return {
    status: "completed",
    stopped: false,
    resumable: false,
    metrics: { best_epoch: 7, total_epochs: 10, test_accuracy: 0.91 },
    tests: [],
    artifacts: { model: "runs/r1/best_model.pt" },
    group: null,
    group_id: "g1",
    task: "classification",
    config: { block: "classification" },
    ...overrides,
  };
}

const ids = (input: NextStepsInput) => nextSteps(input).map((s) => s.id);

describe("nextSteps", () => {
  it("suggests nothing when no rule holds", () => {
    expect(nextSteps(run())).toEqual([]);
  });

  it("suggests nothing while the run is still training", () => {
    expect(nextSteps(run({ status: "running", resumable: true, group_id: null }))).toEqual([]);
  });

  describe("unfinished", () => {
    it("offers to resume a run that stopped short and can continue", () => {
      expect(nextSteps(run({ resumable: true }))).toEqual([{ id: "unfinished", action: "resume" }]);
    });

    it("names a failed run that cannot continue, with no action", () => {
      expect(nextSteps(run({ status: "failed" }))).toEqual([{ id: "unfinished", action: null }]);
    });

    it("counts a user stop as unfinished", () => {
      expect(ids(run({ stopped: true }))).toEqual(["unfinished"]);
    });

    it("suggests nothing else for a run that did not finish", () => {
      const failed = run({
        status: "failed",
        metrics: { best_epoch: 3, total_epochs: 3 },
        group_id: null,
      });
      expect(ids(failed)).toEqual(["unfinished"]);
    });
  });

  describe("stillImproving", () => {
    it("suggests more epochs when the best epoch is the last one", () => {
      const steps = nextSteps(run({ metrics: { best_epoch: 10, total_epochs: 10, test_accuracy: 0.9 } }));
      expect(steps).toEqual([{ id: "stillImproving", action: "reuse" }]);
    });

    it("says nothing about a one-epoch run", () => {
      expect(ids(run({ metrics: { best_epoch: 1, total_epochs: 1, test_accuracy: 0.9 } }))).toEqual([]);
    });

    it("says nothing when the best epoch came before the last (an early stop included)", () => {
      expect(ids(run({ metrics: { best_epoch: 9, total_epochs: 10, test_accuracy: 0.9 } }))).toEqual([]);
    });

    it("says nothing when the epochs were not recorded", () => {
      expect(ids(run({ metrics: { test_accuracy: 0.9 } }))).toEqual([]);
    });
  });

  describe("untested", () => {
    it("asks for a test result when there is neither a test split number nor a test run", () => {
      expect(nextSteps(run({ metrics: { best_epoch: 2, total_epochs: 5, val_r2: 0.8 } }))).toEqual([
        { id: "untested", action: "test" },
      ]);
    });

    it("reads a null test metric as no test result", () => {
      expect(ids(run({ metrics: { test_accuracy: null } }))).toEqual(["untested"]);
    });

    it("is satisfied by a test run on another dataset", () => {
      expect(ids(run({ metrics: {}, tests: [{ test_id: "t1" }] }))).toEqual([]);
    });

    it("is not offered without a checkpoint to test", () => {
      expect(ids(run({ metrics: {}, artifacts: {} }))).toEqual([]);
    });

    it("is not offered for a researcher's task, which the test API refuses", () => {
      expect(ids(run({ metrics: {}, task: "custom:leaf" }))).toEqual([]);
    });
  });

  describe("singleRun", () => {
    it("suggests replicates for a run outside any group", () => {
      expect(nextSteps(run({ group_id: null }))).toEqual([{ id: "singleRun", action: "replicate" }]);
    });

    it("suggests replicates for a run whose config names no block", () => {
      expect(ids(run({ group_id: null, config: {} }))).toEqual(["singleRun"]);
    });

    it("says nothing on a group, which already has its seeds", () => {
      const group = run({ group_id: null, group: { kind: "replicates" }, artifacts: {} });
      expect(ids(group)).toEqual([]);
    });

    it("says nothing on a K-fold, which already measures the spread", () => {
      const cv = run({ group_id: null, metrics: { fold_results: [], test_accuracy: 0.9 } });
      expect(ids(cv)).toEqual([]);
    });

    it("says nothing on a sweep, which is several trainings", () => {
      expect(ids(run({ group_id: null, config: { block: "grid_search" } }))).toEqual([]);
    });
  });

  it("keeps the two highest-priority suggestions when more hold", () => {
    const all = run({
      group_id: null,
      metrics: { best_epoch: 4, total_epochs: 4 },
    });
    expect(nextSteps(all)).toEqual([
      { id: "stillImproving", action: "reuse" },
      { id: "untested", action: "test" },
    ]);
  });

  it("puts the test before the replicates when the run plateaued", () => {
    expect(ids(run({ group_id: null, metrics: { best_epoch: 2, total_epochs: 4 } }))).toEqual([
      "untested",
      "singleRun",
    ]);
  });
});
