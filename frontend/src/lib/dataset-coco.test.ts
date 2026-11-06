import { describe, expect, it } from "vitest";
import {
  canApplyImport,
  initialChoices,
  readySources,
  reasonRows,
  reasonTotal,
  splitConflicts,
  splitsBody,
  undecided,
  type CocoImportPlan,
  type CocoImportSource,
} from "./dataset-coco";

function source(
  name: string,
  detected: CocoImportSource["detected_split"],
): CocoImportSource {
  return {
    name,
    detected_split: detected,
    images_dir: "val2017",
    images_listed: 2,
    images_kept: 2,
    annotations: 3,
    boxes: 3,
    categories: 2,
    skipped_by_reason: {},
    adjusted_by_reason: {},
    listed: [],
  };
}

function plan(over: Partial<CocoImportPlan> = {}): CocoImportPlan {
  return {
    ready: true,
    sources: [
      { name: "a.json", split: "val", detected_split: "val", status: "ready" },
    ],
    unassigned: [],
    per_split: {
      train: { images: 0, boxes: 0 },
      val: { images: 2, boxes: 3 },
      test: { images: 0, boxes: 0 },
    },
    images: 2,
    boxes: 3,
    classes: ["cat"],
    mapping: [],
    renamed: [],
    ...over,
  };
}

describe("initialChoices", () => {
  it("starts from the detected split and leaves the others undecided", () => {
    const choices = initialChoices([
      source("val.json", "val"),
      source("mine.json", null),
    ]);
    expect(choices).toEqual({ "val.json": "val", "mine.json": "" });
  });
});

describe("splitsBody", () => {
  it("sends only the decided files, skip included", () => {
    expect(
      splitsBody({ "a.json": "train", "b.json": "", "c.json": "skip" }),
    ).toEqual({
      "a.json": "train",
      "c.json": "skip",
    });
  });
});

describe("splitConflicts", () => {
  it("names a split chosen twice", () => {
    expect(
      splitConflicts({ "a.json": "train", "b.json": "train", "c.json": "val" }),
    ).toEqual(["train"]);
  });

  it("ignores skipped and undecided files", () => {
    expect(
      splitConflicts({
        "a.json": "skip",
        "b.json": "skip",
        "c.json": "",
        "d.json": "",
      }),
    ).toEqual([]);
  });
});

describe("undecided", () => {
  it("counts the files without a choice", () => {
    expect(undecided({ "a.json": "", "b.json": "val", "c.json": "" })).toBe(2);
  });
});

describe("reasonRows", () => {
  it("orders by count and then by name, and drops zeros", () => {
    expect(
      reasonRows({ crowd: 1, missing_image: 3, bad_bbox: 1, clipped_box: 0 }),
    ).toEqual([
      { reason: "missing_image", count: 3 },
      { reason: "bad_bbox", count: 1 },
      { reason: "crowd", count: 1 },
    ]);
  });

  it("totals the counts", () => {
    expect(reasonTotal({ crowd: 2, missing_image: 3 })).toBe(5);
  });
});

describe("canApplyImport", () => {
  it("needs a ready plan with images, a good name and nothing running", () => {
    expect(canApplyImport(plan(), true, false)).toBe(true);
    expect(canApplyImport(null, true, false)).toBe(false);
    expect(canApplyImport(plan({ ready: false }), true, false)).toBe(false);
    expect(canApplyImport(plan({ images: 0 }), true, false)).toBe(false);
    expect(canApplyImport(plan(), false, false)).toBe(false);
    expect(canApplyImport(plan(), true, true)).toBe(false);
  });
});

describe("readySources", () => {
  it("keeps only the files that feed the result", () => {
    const p = plan({
      sources: [
        {
          name: "a.json",
          split: "val",
          detected_split: "val",
          status: "ready",
        },
        {
          name: "b.json",
          split: null,
          detected_split: null,
          status: "skipped",
        },
        {
          name: "c.json",
          split: null,
          detected_split: null,
          status: "unassigned",
        },
      ],
    });
    expect(readySources(p).map((s) => s.name)).toEqual(["a.json"]);
  });
});
