import { describe, expect, it } from "vitest";
import {
  buildPlanBody,
  customRatios,
  isJobFinished,
  isValidDestName,
  jobPercent,
  openResultForm,
  parsePercent,
  parseSeed,
  planRows,
  ratioLabel,
  reasonRows,
  toggleClass,
  type IncorporatePlan,
} from "./dataset-incorporate";

const ALL = { train: true, val: true, test: true };
const NO_TEST = { train: true, val: true, test: false };

function plan(over: Partial<IncorporatePlan> = {}): IncorporatePlan {
  return {
    seed: 0,
    ratios: { train: 0.7, val: 0.2, test: 0.1 },
    accepted_classes: [],
    classes: ["cat", "dog"],
    added: 3,
    added_per_split: { train: 2, val: 1, test: 0 },
    added_per_class: { train: { cat: 1, dog: 1 }, val: { cat: 1 }, test: {} },
    excluded_count: 3,
    excluded_by_reason: { unreadable: 1, duplicate_existing: 2 },
    excluded: [],
    ...over,
  };
}

describe("jobs", () => {
  it("knows when a job is over", () => {
    expect(isJobFinished("running")).toBe(false);
    for (const s of ["done", "cancelled", "failed"] as const) expect(isJobFinished(s)).toBe(true);
  });

  it("clamps the progress percentage", () => {
    expect(jobPercent(0, 0)).toBe(0);
    expect(jobPercent(1, 3)).toBe(33);
    expect(jobPercent(9, 3)).toBe(100);
  });
});

describe("toggleClass", () => {
  it("adds and removes, keeping the list sorted", () => {
    expect(toggleClass([], "plum")).toEqual(["plum"]);
    expect(toggleClass(["plum"], "bird")).toEqual(["bird", "plum"]);
    expect(toggleClass(["bird", "plum"], "bird")).toEqual(["plum"]);
  });
});

describe("seed and percentages", () => {
  it("accepts only non-negative integers as a seed", () => {
    expect(parseSeed("0")).toBe(0);
    expect(parseSeed(" 42 ")).toBe(42);
    for (const bad of ["", "-1", "1.5", "abc", "99999999999", "2147483648"]) {
      expect(parseSeed(bad)).toBeNull();
    }
  });

  it("parses percentages, comma included, and treats empty as zero", () => {
    expect(parsePercent("")).toBe(0);
    expect(parsePercent("12,5")).toBe(12.5);
    expect(parsePercent("-3")).toBeNull();
    expect(parsePercent("x")).toBeNull();
  });
});

describe("customRatios", () => {
  it("returns the typed values and zeroes a missing split", () => {
    expect(customRatios({ train: "80", val: "20", test: "50" }, NO_TEST)).toEqual({
      train: 80,
      val: 20,
      test: 0,
    });
  });

  it("is null when a box is invalid or everything is zero", () => {
    expect(customRatios({ train: "x", val: "1", test: "1" }, ALL)).toBeNull();
    expect(customRatios({ train: "0", val: "", test: "0" }, ALL)).toBeNull();
  });
});

describe("buildPlanBody", () => {
  const base = {
    accepted: ["bird"],
    seedText: "7",
    custom: false,
    percents: { train: "70", val: "20", test: "10" },
    present: ALL,
  };

  it("omits the ratios for the dataset's own proportion", () => {
    expect(buildPlanBody(base)).toEqual({ accept_classes: ["bird"], seed: 7 });
  });

  it("sends the custom ratios", () => {
    expect(buildPlanBody({ ...base, custom: true })).toEqual({
      accept_classes: ["bird"],
      seed: 7,
      ratios: { train: 70, val: 20, test: 10 },
    });
  });

  it("is null while the seed or the custom ratios are invalid", () => {
    expect(buildPlanBody({ ...base, seedText: "x" })).toBeNull();
    expect(
      buildPlanBody({ ...base, custom: true, percents: { train: "", val: "", test: "" } }),
    ).toBeNull();
  });
});

describe("isValidDestName", () => {
  it("accepts ordinary folder names", () => {
    for (const ok of ["coffee-v2", "my data", "v2.1", "área"]) expect(isValidDestName(ok)).toBe(true);
  });

  it("rejects what the backend rejects", () => {
    for (const bad of ["", "  ", ".", "..", "a/b", "a\\b", "x.", "x ", "con", "NUL.txt", "com1", "a:b", "a*b", "tab\there"]) {
      expect(isValidDestName(bad)).toBe(false);
    }
  });
});

describe("what the preview shows", () => {
  it("labels the ratio over the splits that exist", () => {
    expect(ratioLabel({ train: 0.7, val: 0.2, test: 0.1 }, ALL)).toBe("70 / 20 / 10");
    expect(ratioLabel({ train: 0.8, val: 0.2, test: 0 }, NO_TEST)).toBe("80 / 20");
  });

  it("builds one row per class that receives images", () => {
    expect(planRows(plan())).toEqual([
      { name: "cat", counts: { train: 1, val: 1, test: 0 }, total: 2 },
      { name: "dog", counts: { train: 1, val: 0, test: 0 }, total: 1 },
    ]);
  });

  it("lists exclusion reasons by count", () => {
    expect(reasonRows(plan())).toEqual([
      { reason: "duplicate_existing", count: 2 },
      { reason: "unreadable", count: 1 },
    ]);
  });

  it("opens the result in the same form but at the new folder", () => {
    const form = { task: "detection", base_dir: "/a/ds" };
    expect(openResultForm(form, "/a/ds-v2")).toEqual({ task: "detection", base_dir: "/a/ds-v2" });
    expect(form.base_dir).toBe("/a/ds");
  });
});
