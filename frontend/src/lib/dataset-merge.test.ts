import { describe, expect, it } from "vitest";
import {
  activeHints,
  buildMapping,
  canMerge,
  finalOrder,
  initialRows,
  mergedTargets,
  presentSplits,
  reasonRows,
  rowProblems,
  splitsFromOne,
  targetProblem,
  targetSuggestions,
  updateRow,
  type MergePlan,
  type MergePreview,
} from "./dataset-merge";

const ALL = { train: true, val: true, test: true };

function preview(over: Partial<MergePreview> = {}): MergePreview {
  return {
    suggested_name: "A-B",
    task: "classification",
    a_dir: "/data/A",
    b_dir: "/data/B",
    classes: {
      a: [
        { name: "cat", images: 4 },
        { name: "dog", images: 4 },
      ],
      b: [
        { name: "bird", images: 3 },
        { name: "cat", images: 3 },
      ],
    },
    present: { a: ALL, b: ALL },
    found: { a: 8, b: 6 },
    repeated_within: { a: 0, b: 0 },
    similar_names: [],
    ...over,
  };
}

function plan(over: Partial<MergePlan> = {}): MergePlan {
  return {
    classes: [],
    background: { train: 0, val: 0, test: 0 },
    splits: {
      train: { present: true, images: 5, from_a: 3, from_b: 2 },
      val: { present: true, images: 1, from_a: 1, from_b: 0 },
      test: { present: false, images: 0, from_a: 0, from_b: 0 },
    },
    only_in: { a: ["val"], b: [] },
    images_total: 6,
    dropped_count: 3,
    dropped_by_reason: { discarded_class: 1, all_boxes_discarded: 2 },
    dropped: [],
    duplicates_dropped: 0,
    duplicates_cross_split: 0,
    duplicates: [],
    ...over,
  };
}

describe("initialRows", () => {
  it("lists every class of A then of B, each keeping its own name", () => {
    const rows = initialRows(preview());
    expect(rows.map((r) => `${r.side}:${r.name}`)).toEqual(["a:cat", "a:dog", "b:bird", "b:cat"]);
    expect(rows.every((r) => r.target === r.name && !r.discard)).toBe(true);
    expect(rows[0].images).toBe(4);
  });
});

describe("updateRow and buildMapping", () => {
  it("changes only the row asked for", () => {
    const rows = updateRow(initialRows(preview()), "b", "cat", { target: "feline" });
    expect(rows.find((r) => r.side === "b" && r.name === "cat")?.target).toBe("feline");
    expect(rows.find((r) => r.side === "a" && r.name === "cat")?.target).toBe("cat");
  });

  it("sends every class, a discarded one as null and targets trimmed", () => {
    let rows = updateRow(initialRows(preview()), "a", "dog", { discard: true });
    rows = updateRow(rows, "b", "bird", { target: " pet " });
    expect(buildMapping(rows)).toEqual({
      a: { cat: "cat", dog: null },
      b: { bird: "pet", cat: "cat" },
    });
  });
});

describe("target checks", () => {
  const row = (target: string, discard = false) => ({
    side: "a" as const,
    name: "cat",
    images: 1,
    target,
    discard,
  });

  it("rejects empty targets, but not on a discarded row", () => {
    expect(targetProblem(row("  "), "classification")).toBe("empty");
    expect(targetProblem(row("", true), "classification")).toBeNull();
  });

  it("applies the folder-name rule to classification only", () => {
    expect(targetProblem(row("a/b"), "classification")).toBe("invalid");
    expect(targetProblem(row("a/b"), "detection")).toBeNull();
    expect(targetProblem(row("two\nlines"), "detection")).toBe("invalid");
  });

  it("flags targets that differ only by case in classification", () => {
    const rows = updateRow(
      updateRow(initialRows(preview()), "a", "dog", { target: "Cat" }),
      "b",
      "bird",
      { target: "Bird" },
    );
    const problems = rowProblems(rows, "classification");
    expect(problems.get("a:dog")).toBe("case");
    expect(problems.get("a:cat")).toBe("case");
    expect(problems.has("b:bird")).toBe(false);
    expect(rowProblems(rows, "detection").size).toBe(0);
  });

  it("does not flag the same target used twice: that is a merge", () => {
    expect(rowProblems(initialRows(preview()), "classification").size).toBe(0);
  });
});

describe("the final classes", () => {
  it("are A's targets in order, then B's new ones", () => {
    expect(finalOrder(initialRows(preview()))).toEqual(["cat", "dog", "bird"]);
    const rows = updateRow(initialRows(preview()), "a", "cat", { discard: true });
    // A's cat is discarded, but B's cat is a target of its own and comes after B's earlier class.
    expect(finalOrder(rows)).toEqual(["dog", "bird", "cat"]);
  });

  it("knows which targets are merged", () => {
    expect([...mergedTargets(initialRows(preview()))]).toEqual(["cat"]);
    const rows = updateRow(initialRows(preview()), "b", "cat", { discard: true });
    expect(mergedTargets(rows).size).toBe(0);
  });

  it("offers the names of both datasets once, sorted", () => {
    expect(targetSuggestions(preview())).toEqual(["bird", "cat", "dog"]);
  });
});

describe("hints", () => {
  const similar = preview({
    classes: {
      a: [{ name: "cat", images: 1 }],
      b: [{ name: "Cat", images: 1 }],
    },
    similar_names: [["Cat", "cat"]],
  });

  it("shows a hint while both spellings are still targets", () => {
    expect(activeHints(similar, initialRows(similar))).toEqual([["Cat", "cat"]]);
  });

  it("drops the hint once the user made them one name", () => {
    const rows = updateRow(initialRows(similar), "b", "Cat", { target: "cat" });
    expect(activeHints(similar, rows)).toEqual([]);
  });
});

describe("plan numbers", () => {
  it("lists splits from one dataset only", () => {
    expect(splitsFromOne(plan())).toEqual([["a", "val"]]);
  });

  it("orders drop reasons by count", () => {
    expect(reasonRows(plan()).map((r) => r.reason)).toEqual([
      "all_boxes_discarded",
      "discarded_class",
    ]);
  });

  it("lists the splits the result has", () => {
    expect(presentSplits(plan())).toEqual(["train", "val"]);
  });

  it("allows applying only with a plan, images and no row problem", () => {
    const none = new Map();
    expect(canMerge(none, plan())).toBe(true);
    expect(canMerge(none, null)).toBe(false);
    expect(canMerge(none, plan({ images_total: 0 }))).toBe(false);
    expect(canMerge(new Map([["a:cat", "empty" as const]]), plan())).toBe(false);
  });
});
