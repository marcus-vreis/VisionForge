import { describe, expect, it } from "vitest";
import type { ExplorerSummary } from "./dataset-explorer";
import {
  LAB_SECTIONS,
  canOpenLab,
  classStacks,
  cocoModes,
  datasetName,
  initialSection,
  labSections,
  middleTruncate,
  stepSection,
  summaryNotes,
  totalBoxes,
  trapTarget,
} from "./dataset-lab";

const split = (images: number, per_class: Record<string, number>, present = true) => ({
  present,
  images,
  per_class,
  boxes: {},
  unlabeled: 0,
});

function summary(over: Partial<ExplorerSummary> = {}): ExplorerSummary {
  return {
    base_dir: "/data/cafe",
    task: "classification",
    class_names: ["a", "b"],
    total_images: 10,
    splits: {
      train: split(6, { a: 4, b: 2 }),
      val: split(2, { a: 1, b: 1 }),
      test: split(2, { a: 1, b: 1 }),
    },
    ...over,
  };
}

describe("lab sections", () => {
  it("offers every section, in the rail order, for both dataset types", () => {
    expect(LAB_SECTIONS).toEqual(["overview", "resplit", "incorporate", "merge", "coco"]);
    expect(labSections("classification")).toEqual([...LAB_SECTIONS]);
    expect(labSections("detection")).toEqual([...LAB_SECTIONS]);
  });

  it("offers only COCO without an opened dataset", () => {
    expect(labSections(null)).toEqual(["coco"]);
  });

  it("returns a copy the caller cannot use to change the rail", () => {
    const s = labSections("detection");
    s.pop();
    expect(labSections("detection")).toHaveLength(5);
  });

  it("exports to COCO only from a detection dataset; import is always there", () => {
    expect(cocoModes("detection")).toEqual(["export", "import"]);
    expect(cocoModes("classification")).toEqual(["import"]);
    expect(cocoModes(null)).toEqual(["import"]);
  });

  it("starts on the wanted section when offered, else on the first", () => {
    expect(initialSection("classification")).toBe("overview");
    expect(initialSection("detection", "merge")).toBe("merge");
    expect(initialSection(null)).toBe("coco");
    expect(initialSection(null, "overview")).toBe("coco");
  });

  it("moves through the rail with the arrow keys, Home and End, wrapping", () => {
    const all = labSections("classification");
    expect(stepSection(all, "overview", "ArrowDown")).toBe("resplit");
    expect(stepSection(all, "overview", "ArrowRight")).toBe("resplit");
    expect(stepSection(all, "overview", "ArrowUp")).toBe("coco");
    expect(stepSection(all, "coco", "ArrowDown")).toBe("overview");
    expect(stepSection(all, "merge", "ArrowLeft")).toBe("incorporate");
    expect(stepSection(all, "merge", "Home")).toBe("overview");
    expect(stepSection(all, "overview", "End")).toBe("coco");
    expect(stepSection(all, "overview", "Enter")).toBeNull();
    expect(stepSection([], "overview", "ArrowDown")).toBeNull();
  });
});

describe("focus trap", () => {
  it("wraps from the last element to the first and back", () => {
    expect(trapTarget(4, 5, false)).toBe(0);
    expect(trapTarget(0, 5, true)).toBe(4);
  });

  it("leaves a move inside the dialog to the browser", () => {
    expect(trapTarget(2, 5, false)).toBeNull();
    expect(trapTarget(2, 5, true)).toBeNull();
  });

  it("pulls a focus that escaped back to the nearest edge", () => {
    expect(trapTarget(-1, 5, false)).toBe(0);
    expect(trapTarget(-1, 5, true)).toBe(4);
  });

  it("does nothing when nothing can take the focus", () => {
    expect(trapTarget(-1, 0, false)).toBeNull();
  });
});

describe("path display", () => {
  it("keeps a short path whole", () => {
    expect(middleTruncate("C:/data/cafe", 40)).toBe("C:/data/cafe");
  });

  it("cuts a long path in the middle, to the exact length", () => {
    const out = middleTruncate("C:/datasets/projetos/2024/cafe/folhas", 20);
    expect(out).toHaveLength(20);
    expect(out).toBe("C:/dataset…fe/folhas");
  });

  it("names the dataset after its last folder, on any separator", () => {
    expect(datasetName("C:\\datasets\\cafe\\")).toBe("cafe");
    expect(datasetName("/home/u/data/folhas")).toBe("folhas");
    expect(datasetName("  cafe  ")).toBe("cafe");
  });
});

describe("class stacks", () => {
  it("stacks each class by split, largest first, widths relative to the largest", () => {
    const rows = classStacks(summary());
    expect(rows.map((r) => r.name)).toEqual(["a", "b"]);
    expect(rows[0].counts).toEqual({ train: 4, val: 1, test: 1 });
    expect(rows[0].total).toBe(6);
    expect(rows[0].pct.train + rows[0].pct.val + rows[0].pct.test).toBeCloseTo(100);
    expect(rows[1].total).toBe(4);
    expect(rows[1].pct.train).toBeCloseTo((2 / 6) * 100);
  });

  it("counts boxes when asked and survives an empty dataset", () => {
    const s = summary({ task: "detection" });
    s.splits.train.boxes = { a: 7 };
    expect(classStacks(s, "boxes")[0]).toMatchObject({ name: "a", total: 7 });
    const empty = summary({ total_images: 0, splits: {
      train: split(0, {}), val: split(0, {}), test: split(0, {}),
    } });
    expect(classStacks(empty).every((r) => r.pct.train === 0)).toBe(true);
    expect(totalBoxes(s)).toBe(7);
  });
});

describe("summary notes", () => {
  it("says nothing about a complete dataset", () => {
    expect(summaryNotes(summary())).toEqual([]);
    expect(canOpenLab(summary())).toBe(true);
  });

  it("flags an empty dataset alone and keeps the lab closed", () => {
    const s = summary({ total_images: 0, splits: {
      train: split(0, {}, false), val: split(0, {}, false), test: split(0, {}, false),
    } });
    expect(summaryNotes(s)).toEqual([{ kind: "empty" }]);
    expect(canOpenLab(s)).toBe(false);
    expect(canOpenLab(null)).toBe(false);
  });

  it("flags a missing train or validation folder, not a missing test", () => {
    const s = summary();
    s.splits.val = split(0, {}, false);
    s.splits.test = split(0, {}, false);
    expect(summaryNotes(s)).toEqual([{ kind: "missing_split", split: "val" }]);
  });

  it("counts detection images without boxes over the present splits", () => {
    const s = summary({ task: "detection" });
    s.splits.train.unlabeled = 3;
    s.splits.test = { ...split(0, {}, false), unlabeled: 9 };
    expect(summaryNotes(s)).toEqual([{ kind: "unlabeled", count: 3 }]);
  });
});
