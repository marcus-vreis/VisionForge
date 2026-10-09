import { describe, expect, it } from "vitest";

import type {
  GroupVariant,
  MetricAggregate,
  RunGroup,
  RunGroupBrief,
  RunSummary,
} from "../types/run";
import {
  aggregateForRow,
  bestByMean,
  childTarget,
  ciHalfWidth,
  fewestSeeds,
  foldGroups,
  formatAggregate,
  formatNumber,
  formatP,
  hasFewSeeds,
  holmVerdict,
  isGroupRun,
} from "./run-groups";

function run(id: string, extra: Partial<RunSummary> = {}): RunSummary {
  return {
    run_id: id,
    experiment_name: id,
    model_arch: "resnet18",
    task: "regression",
    status: "completed",
    started_at: "2026-01-01T00:00:00",
    finished_at: null,
    epochs_completed: 1,
    final_metrics: {},
    ...extra,
  };
}

function brief(childIds: string[]): RunGroupBrief {
  return {
    kind: "replicates",
    metric: "r2",
    seeds: [1, 2],
    n_requested: 2,
    n_finished: 2,
    stopped: false,
    child_ids: childIds,
    final_aggregates: {},
    variants: [],
    best_by_mean: null,
  };
}

const AGG: MetricAggregate = {
  n: 5,
  mean: 0.8123456,
  std: 0.02,
  std_ddof: 1,
  min: 0.78,
  max: 0.84,
  ci95_low: 0.79,
  ci95_high: 0.83,
};

describe("foldGroups", () => {
  it("lists a group as one entry with its seeds folded under it", () => {
    const runs = [
      run("g1", { group: brief(["a", "b"]) }),
      run("b", { group_id: "g1" }),
      run("a", { group_id: "g1" }),
      run("solo"),
    ];
    const entries = foldGroups(runs);
    expect(entries.map((e) => e.run.run_id)).toEqual(["g1", "solo"]);
    // The seeds follow the order the group recorded, not the list's.
    expect(entries[0].children.map((c) => c.run_id)).toEqual(["a", "b"]);
    expect(entries[1].children).toEqual([]);
  });

  it("keeps a seed visible when its group is gone from the list", () => {
    // Deleting a group removes only its summary; the seeds keep a dangling id.
    const entries = foldGroups([run("a", { group_id: "deleted" }), run("b")]);
    expect(entries.map((e) => e.run.run_id)).toEqual(["a", "b"]);
  });

  it("keeps the order of the list for the top level", () => {
    const entries = foldGroups([run("new"), run("g", { group: brief([]) }), run("old")]);
    expect(entries.map((e) => e.run.run_id)).toEqual(["new", "g", "old"]);
  });

  it("does not fold a group into another one", () => {
    const entries = foldGroups([
      run("outer", { group: brief([]) }),
      run("inner", { group: brief([]), group_id: "outer" }),
    ]);
    expect(entries.map((e) => e.run.run_id)).toEqual(["outer", "inner"]);
  });

  it("treats a list with no groups as it was", () => {
    const entries = foldGroups([run("a"), run("b")]);
    expect(entries.map((e) => e.run.run_id)).toEqual(["a", "b"]);
    expect(entries.every((e) => e.children.length === 0)).toBe(true);
  });
});

describe("isGroupRun", () => {
  it("is true only for a run that carries a group", () => {
    expect(isGroupRun(run("g", { group: brief([]) }))).toBe(true);
    expect(isGroupRun(run("a", { group_id: "g" }))).toBe(false);
    expect(isGroupRun(run("a", { group: null }))).toBe(false);
  });
});

describe("ciHalfWidth", () => {
  it("is half the interval the report gave", () => {
    expect(ciHalfWidth(AGG)).toBeCloseTo(0.02, 10);
  });

  it("is null when there is no interval, as with one seed", () => {
    expect(ciHalfWidth({ ci95_low: null, ci95_high: null })).toBeNull();
    expect(ciHalfWidth(null)).toBeNull();
    expect(ciHalfWidth({ ci95_low: Number.NaN, ci95_high: 1 })).toBeNull();
  });
});

describe("formatAggregate", () => {
  it("prints the mean, the half-width, the bounds and the n", () => {
    expect(formatAggregate(AGG)).toEqual({
      mean: "0.8123",
      half: "0.0200",
      low: "0.7900",
      high: "0.8300",
      std: "0.0200",
      n: 5,
    });
  });

  it("prints a lone seed as a mean with no interval or spread", () => {
    const one = { ...AGG, n: 1, std: null, ci95_low: null, ci95_high: null };
    expect(formatAggregate(one)).toEqual({
      mean: "0.8123",
      half: null,
      low: null,
      high: null,
      std: null,
      n: 1,
    });
  });

  it("has nothing to print without a mean", () => {
    expect(formatAggregate({ ...AGG, mean: null })).toBeNull();
    expect(formatAggregate(undefined)).toBeNull();
  });
});

describe("aggregateForRow", () => {
  const group: Pick<RunGroup, "metric_keys" | "aggregates"> = {
    metric_keys: { test_r2: "r2" },
    aggregates: { r2: AGG, rmse: { ...AGG, mean: 1 } },
  };

  it("finds the aggregate the server says filled the row", () => {
    expect(aggregateForRow(group, "test_r2")).toBe(AGG);
  });

  it("gives a validation-score row no interval of the test split", () => {
    // `r2` is the best-epoch validation row; the group's r2 is the test split's.
    expect(aggregateForRow(group, "r2")).toBeNull();
  });

  it("is null for a row no aggregate fills, and for no group", () => {
    expect(aggregateForRow(group, "total_epochs")).toBeNull();
    expect(aggregateForRow(null, "test_r2")).toBeNull();
    expect(aggregateForRow({}, "test_r2")).toBeNull();
  });
});

describe("bestByMean", () => {
  it("names the best variant when the server did", () => {
    expect(bestByMean({ best_by_mean: "lr", not_run: [] })).toEqual({
      kind: "best",
      label: "lr",
    });
  });

  it("says a stop is why, when variants never ran", () => {
    expect(bestByMean({ best_by_mean: null, not_run: ["wd"] })).toEqual({
      kind: "none",
      reason: "stopped",
    });
  });

  it("otherwise blames the seeds the variants share", () => {
    expect(bestByMean({ best_by_mean: null, not_run: [] })).toEqual({
      kind: "none",
      reason: "too-few-seeds",
    });
    expect(bestByMean({ best_by_mean: null })).toEqual({
      kind: "none",
      reason: "too-few-seeds",
    });
  });
});

describe("formatNumber", () => {
  it("prints four decimals and a dash for no value", () => {
    expect(formatNumber(0.12345)).toBe("0.1235");
    expect(formatNumber(2, 1)).toBe("2.0");
    expect(formatNumber(null)).toBe("—");
    expect(formatNumber(Number.POSITIVE_INFINITY)).toBe("—");
  });
});

describe("formatP", () => {
  it("prints four decimals", () => {
    expect(formatP(0.0312)).toBe("0.0312");
    expect(formatP(0.5)).toBe("0.5000");
  });

  it("does not print a zero it did not measure", () => {
    expect(formatP(0.00001)).toBe("<0.0001");
  });

  it("prints a dash for no value", () => {
    expect(formatP(null)).toBe("—");
    expect(formatP(Number.NaN)).toBe("—");
    expect(formatP(undefined)).toBe("—");
  });
});

describe("holmVerdict", () => {
  it("reads the server's flag and does not look at the raw p", () => {
    expect(holmVerdict({ significant: true })).toBe("yes");
    expect(holmVerdict({ significant: false })).toBe("no");
  });
});

describe("childTarget", () => {
  const known = new Set(["a"]);

  it("opens a seed that is still in History", () => {
    expect(childTarget({ run_id: "a" }, known)).toBe("a");
  });

  it("does not link a seed that was deleted or never got a run", () => {
    expect(childTarget({ run_id: "gone" }, known)).toBeNull();
    expect(childTarget({ run_id: null }, known)).toBeNull();
  });

  it("links without checking when the list is unknown", () => {
    expect(childTarget({ run_id: "a" }, undefined)).toBe("a");
  });
});

describe("hasFewSeeds", () => {
  it("warns below five seeds", () => {
    expect(hasFewSeeds(2)).toBe(true);
    expect(hasFewSeeds(4)).toBe(true);
    expect(hasFewSeeds(5)).toBe(false);
  });
});

describe("fewestSeeds", () => {
  const variant = (n: number | null, successful: number | null = null): GroupVariant => ({
    overrides: {},
    aggregates: n === null ? {} : { r2: { ...AGG, n } },
    successful,
    seeds_finished: [],
    children: [],
  });

  it("is the smallest seed count among the variants that trained", () => {
    expect(fewestSeeds({ metric: "r2", variants: { a: variant(6), b: variant(3) } })).toBe(3);
  });

  it("falls back to the variant's own success count when it has no aggregate", () => {
    expect(fewestSeeds({ metric: "r2", variants: { a: variant(6), b: variant(null, 2) } })).toBe(2);
  });

  it("skips a variant nothing finished: that is not a seed count", () => {
    expect(fewestSeeds({ metric: "r2", variants: { a: variant(null, 0), b: variant(4) } })).toBe(4);
  });

  it("is null when no variant has a count, or there are no variants", () => {
    expect(fewestSeeds({ metric: "r2", variants: { a: variant(null, 0) } })).toBeNull();
    expect(fewestSeeds({ metric: "r2", variants: {} })).toBeNull();
    expect(fewestSeeds({ metric: "r2" })).toBeNull();
  });

  it("feeds hasFewSeeds: a comparison of three seeds each warns, five does not", () => {
    const few = fewestSeeds({ metric: "r2", variants: { a: variant(3), b: variant(3) } });
    const enough = fewestSeeds({ metric: "r2", variants: { a: variant(5), b: variant(5) } });
    expect(few !== null && hasFewSeeds(few)).toBe(true);
    expect(enough !== null && hasFewSeeds(enough)).toBe(false);
  });
});
