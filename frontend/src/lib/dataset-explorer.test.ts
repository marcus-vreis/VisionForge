import { describe, expect, it } from "vitest";
import {
  barsOf,
  binBars,
  binLabel,
  classSplitRows,
  explorerBody,
  hasProblems,
  isScanFinished,
  makeDefaultExplorerForm,
  pagingOf,
  problemCounts,
  scanPercent,
  stepOffset,
  thumbUrl,
  type ExplorerSummary,
  type ScanReport,
} from "./dataset-explorer";

const emptySplit = { present: true, images: 0, per_class: {}, boxes: {}, unlabeled: 0 };

function report(over: Partial<ScanReport> = {}): ScanReport {
  return {
    total_images: 10,
    scanned: 10,
    limit: 20000,
    sampled: false,
    sample_step: 1,
    cancelled: false,
    complete: true,
    corrupted_count: 0,
    corrupted: [],
    duplicate_group_count: 0,
    duplicate_file_count: 0,
    duplicate_groups: [],
    leakage_group_count: 0,
    leakage_groups: [],
    sizes: { count: 0 },
    ...over,
  };
}

describe("explorerBody", () => {
  it("sends the split folders for classification, trimmed base", () => {
    const form = { ...makeDefaultExplorerForm(), base_dir: "  C:/d  ", val_dir: "valid" };
    expect(explorerBody(form)).toEqual({
      base_dir: "C:/d",
      task: "classification",
      train_dir: "train",
      val_dir: "valid",
      test_dir: "test",
    });
  });

  it("leaves the split folders out for detection", () => {
    const form = { ...makeDefaultExplorerForm(), task: "detection" as const, base_dir: "C:/y" };
    expect(explorerBody(form)).toEqual({ base_dir: "C:/y", task: "detection" });
  });

  it("falls back to the default name when a folder is blank", () => {
    const form = { ...makeDefaultExplorerForm(), base_dir: "x", train_dir: "" };
    expect(explorerBody(form).train_dir).toBe("train");
  });
});

describe("paging", () => {
  it("describes an empty gallery", () => {
    expect(pagingOf(0, 48, 0)).toEqual({ page: 0, pages: 0, from: 0, to: 0 });
  });

  it("describes the last, partial page", () => {
    expect(pagingOf(100, 48, 96)).toEqual({ page: 2, pages: 3, from: 97, to: 100 });
  });

  it("steps and clamps the offset", () => {
    expect(stepOffset(100, 48, 0, 1)).toBe(48);
    expect(stepOffset(100, 48, 96, 1)).toBe(96);
    expect(stepOffset(100, 48, 0, -1)).toBe(0);
    expect(stepOffset(0, 48, 0, 1)).toBe(0);
  });
});

describe("barsOf", () => {
  it("sorts descending with widths relative to the largest", () => {
    const bars = barsOf({ b: 5, a: 10, c: 5 });
    expect(bars.map((b) => b.label)).toEqual(["a", "b", "c"]);
    expect(bars.map((b) => b.pct)).toEqual([100, 50, 50]);
  });

  it("does not divide by zero", () => {
    expect(barsOf({ a: 0 })[0].pct).toBe(0);
    expect(barsOf({})).toEqual([]);
  });
});

describe("classSplitRows", () => {
  it("fills a zero where a class is absent from a split", () => {
    const summary: ExplorerSummary = {
      base_dir: "x",
      task: "classification",
      class_names: ["cat", "dog"],
      total_images: 3,
      splits: {
        train: { ...emptySplit, per_class: { cat: 2, dog: 1 } },
        val: { ...emptySplit, per_class: { cat: 1 } },
        test: { ...emptySplit, present: false },
      },
    };
    const rows = classSplitRows(summary, "per_class");
    expect(rows[0]).toEqual({ name: "cat", counts: { train: 2, val: 1, test: 0 }, total: 3 });
    expect(rows[1].counts).toEqual({ train: 1, val: 0, test: 0 });
  });
});

describe("scan progress and problems", () => {
  it("computes a clamped percentage", () => {
    expect(scanPercent(0, 0)).toBe(0);
    expect(scanPercent(1, 3)).toBe(33);
    expect(scanPercent(5, 3)).toBe(100);
  });

  it("knows which states are final", () => {
    expect(isScanFinished("running")).toBe(false);
    for (const s of ["done", "cancelled", "failed"] as const) {
      expect(isScanFinished(s)).toBe(true);
    }
  });

  it("counts findings and detects a clean report", () => {
    expect(hasProblems(report())).toBe(false);
    const dirty = report({ corrupted_count: 2, leakage_group_count: 1 });
    expect(hasProblems(dirty)).toBe(true);
    expect(problemCounts(dirty)).toEqual({ corrupted: 2, duplicates: 0, leakage: 1 });
  });
});

describe("bins", () => {
  it("labels closed and open ranges", () => {
    expect(binLabel({ lo: 64, hi: 128, count: 1 })).toBe("64-128");
    expect(binLabel({ lo: 2048, hi: null, count: 1 })).toBe("2048+");
    expect(binLabel({ lo: 0.9, hi: 1.1, count: 1 }, 2)).toBe("0.90-1.10");
  });

  it("scales bars to the fullest bin", () => {
    const bars = binBars([
      { lo: 0, hi: 64, count: 2 },
      { lo: 64, hi: null, count: 4 },
    ]);
    expect(bars.map((b) => b.pct)).toEqual([50, 100]);
  });
});

describe("thumbUrl", () => {
  it("encodes the root and the relative path", () => {
    const url = thumbUrl("C:/a b/c&d", "train/x#1.png", 160, null);
    expect(url).toContain("root=C%3A%2Fa+b%2Fc%26d");
    expect(url).toContain("path=train%2Fx%231.png");
    expect(url).toContain("size=160");
    expect(url).not.toContain("label_path");
  });

  it("adds the label path for a detection image", () => {
    expect(thumbUrl("r", "images/a.png", 96, "labels/a.txt")).toContain(
      "label_path=labels%2Fa.txt",
    );
  });
});
