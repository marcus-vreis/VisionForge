import { describe, expect, it } from "vitest";

import { seedNote } from "./compare-seeds";

describe("seedNote", () => {
  it("warns when two runs differ in a metric that has a direction", () => {
    expect(seedNote([{ direction: "higher", values: [0.9, 0.92] }])).toBe("single-seed");
  });

  it("warns for a lower-is-better metric too", () => {
    expect(seedNote([{ direction: "lower", values: [0.31, 0.29] }])).toBe("single-seed");
  });

  it("warns once any row shows a gap, whatever the other rows say", () => {
    const rows = [
      { direction: null, values: [3, 9] },
      { direction: "higher" as const, values: [0.5, 0.5] },
      { direction: "lower" as const, values: [0.3, 0.2, 0.4] },
    ];
    expect(seedNote(rows)).toBe("single-seed");
  });

  it("says nothing when every metric is equal: no gap is on screen", () => {
    expect(seedNote([{ direction: "higher", values: [0.9, 0.9] }])).toBeNull();
  });

  it("says nothing about a count or a decision point, which have no better end", () => {
    expect(seedNote([{ direction: null, values: [12, 10] }])).toBeNull();
  });

  it("says nothing when only one run measured the metric", () => {
    expect(seedNote([{ direction: "higher", values: [0.9, null] }])).toBeNull();
  });

  it("says nothing for a table with no rows", () => {
    expect(seedNote([])).toBeNull();
  });

  it("still warns when a run lacks a value but two others differ", () => {
    expect(seedNote([{ direction: "higher", values: [null, 0.7, 0.8] }])).toBe("single-seed");
  });
});
