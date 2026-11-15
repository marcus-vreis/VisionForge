import { describe, expect, it } from "vitest";
import {
  buildBody,
  compareRows,
  currentPercents,
  customRatios,
  droppedReasonRows,
  effectiveLock,
  formatShare,
  ratioRows,
  vanishedSplits,
  type ResplitPlan,
} from "./dataset-resplit";

const ALL = { train: true, val: true, test: true };
const NO_TEST = { train: true, val: true, test: false };
const TEXTS = { train: "70", val: "20", test: "10" };

function plan(over: Partial<ResplitPlan> = {}): ResplitPlan {
  return {
    seed: 0,
    ratios: { train: 0.6, val: 0.4, test: 0 },
    achieved: { train: 0.6, val: 0.4 },
    locked_splits: ["test"],
    by_group: false,
    before: { train: { cat: 8, dog: 8 }, val: { cat: 2 }, test: { cat: 2, dog: 2 } },
    after: { train: { cat: 6, dog: 6 }, val: { cat: 4, dog: 2 }, test: { cat: 2, dog: 2 } },
    before_totals: { train: 16, val: 2, test: 4 },
    after_totals: { train: 12, val: 6, test: 4 },
    kept: 22,
    moved: 4,
    dropped_count: 3,
    dropped_by_reason: { group_in_test: 1, duplicate_of_test: 2 },
    dropped: [],
    duplicate_sets: 0,
    duplicate_extras: 0,
    duplicate_sets_split_before: 0,
    warnings: [],
    ...over,
  };
}

describe("lock", () => {
  it("only holds when the dataset has a test split", () => {
    expect(effectiveLock(true, ALL)).toBe(true);
    expect(effectiveLock(true, NO_TEST)).toBe(false);
    expect(effectiveLock(false, ALL)).toBe(false);
  });
});

describe("customRatios", () => {
  it("reads the boxes and sends a locked test as zero", () => {
    expect(customRatios(TEXTS, true)).toEqual({ train: 70, val: 20, test: 0 });
    expect(customRatios(TEXTS, false)).toEqual({ train: 70, val: 20, test: 10 });
  });
  it("rejects text and all-zero ratios", () => {
    expect(customRatios({ ...TEXTS, val: "x" }, true)).toBeNull();
    expect(customRatios({ train: "0", val: "", test: "5" }, true)).toBeNull();
  });
  it("accepts a comma as the decimal mark", () => {
    expect(customRatios({ train: "7,5", val: "2,5", test: "0" }, true)).toEqual({
      train: 7.5,
      val: 2.5,
      test: 0,
    });
  });
});

describe("buildBody", () => {
  const base = {
    seedText: "3",
    lockTest: true,
    present: ALL,
    custom: false,
    percents: TEXTS,
    byGroup: null,
  };
  it("omits ratios and by_group when the user chose neither", () => {
    expect(buildBody(base)).toEqual({ seed: 3, lock_test: true });
  });
  it("sends the lock flag as the user set it", () => {
    expect(buildBody({ ...base, lockTest: false })).toEqual({ seed: 3, lock_test: false });
  });
  it("sends by_group once a file was loaded", () => {
    expect(buildBody({ ...base, byGroup: false })).toEqual({
      seed: 3,
      lock_test: true,
      by_group: false,
    });
  });
  it("sends custom ratios", () => {
    expect(buildBody({ ...base, custom: true })).toEqual({
      seed: 3,
      lock_test: true,
      ratios: { train: 70, val: 20, test: 0 },
    });
  });
  it("is null for a bad seed or bad custom ratios", () => {
    expect(buildBody({ ...base, seedText: "-1" })).toBeNull();
    expect(buildBody({ ...base, custom: true, percents: { ...TEXTS, train: "a" } })).toBeNull();
  });
});

describe("currentPercents", () => {
  it("uses the pooled splits only when the test is locked", () => {
    expect(currentPercents({ train: 16, val: 4, test: 4 }, true)).toEqual({
      train: "80",
      val: "20",
      test: "0",
    });
  });
  it("includes the test when it is not locked", () => {
    expect(currentPercents({ train: 16, val: 4, test: 4 }, false)).toEqual({
      train: "67",
      val: "17",
      test: "17",
    });
  });
  it("does not divide by zero", () => {
    expect(currentPercents({ train: 0, val: 0, test: 0 }, true)).toEqual({
      train: "0",
      val: "0",
      test: "0",
    });
  });
});

describe("formatShare", () => {
  it("keeps one decimal at most", () => {
    expect(formatShare(0.6)).toBe("60");
    expect(formatShare(1 / 3)).toBe("33.3");
    expect(formatShare(0)).toBe("0");
  });
});

describe("ratioRows", () => {
  it("lists requested and achieved for the distributed splits only", () => {
    expect(ratioRows(plan())).toEqual([
      { split: "train", requested: 0.6, achieved: 0.6 },
      { split: "val", requested: 0.4, achieved: 0.4 },
    ]);
  });
  it("includes the test when it is not locked", () => {
    const rows = ratioRows(
      plan({
        locked_splits: [],
        ratios: { train: 0.5, val: 0.25, test: 0.25 },
        achieved: { train: 0.5, val: 0.25, test: 0.25 },
      }),
    );
    expect(rows.map((r) => r.split)).toEqual(["train", "val", "test"]);
  });
});

describe("compareRows", () => {
  it("has one row per class with counts before and after in every split", () => {
    const rows = compareRows(plan());
    expect(rows.map((r) => r.name)).toEqual(["cat", "dog"]);
    expect(rows[1].before).toEqual({ train: 8, val: 0, test: 2 });
    expect(rows[1].after).toEqual({ train: 6, val: 2, test: 2 });
  });
});

describe("droppedReasonRows", () => {
  it("sorts the most frequent first", () => {
    expect(droppedReasonRows(plan())).toEqual([
      { reason: "duplicate_of_test", count: 2 },
      { reason: "group_in_test", count: 1 },
    ]);
  });
});

describe("vanishedSplits", () => {
  it("names the existing splits that end up empty", () => {
    const p = plan({ after_totals: { train: 18, val: 0, test: 4 } });
    expect(vanishedSplits(p, ALL)).toEqual(["val"]);
    expect(vanishedSplits(p, { train: true, val: false, test: true })).toEqual([]);
  });
});
