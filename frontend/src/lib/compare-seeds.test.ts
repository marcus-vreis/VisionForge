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

describe("seedNote with replicate groups (ADR-113)", () => {
  // `values` are every run's value, group means included: they are the ones the
  // table highlights an extreme among, so they are the ones the note is judged on.
  it("is silent when every compared run is a group: each cell carries its own interval", () => {
    expect(seedNote([{ direction: "higher", values: [0.9, 0.8] }], [true, true])).toBeNull();
  });

  it("warns for one single run beside a group mean: the highlighted extreme involves a seed", () => {
    // The group is the highest cell, the single run the lowest: either way the
    // table points at a gap that rests on one seed.
    expect(seedNote([{ direction: "higher", values: [0.8, 0.9] }], [false, true])).toBe(
      "single-seed-mixed",
    );
    expect(seedNote([{ direction: "higher", values: [0.9, 0.8] }], [false, true])).toBe(
      "single-seed-mixed",
    );
  });

  it("speaks of the single runs only when a group is in the selection too", () => {
    expect(seedNote([{ direction: "higher", values: [0.9, 0.8, 0.85] }], [false, false, true])).toBe(
      "single-seed-mixed",
    );
  });

  it("is silent when the single run did not measure the metric: only means are on the row", () => {
    expect(seedNote([{ direction: "higher", values: [null, 0.8, 0.9] }], [false, true, true])).toBeNull();
  });

  it("is silent where the table highlights nothing, groups or not", () => {
    expect(seedNote([{ direction: "higher", values: [0.9, 0.9] }], [false, true])).toBeNull();
    expect(seedNote([{ direction: null, values: [0.9, 0.8] }], [false, true])).toBeNull();
  });

  it("warns on a row that has a single run even when another row is groups only", () => {
    const rows = [
      { direction: "higher" as const, values: [null, 0.8, 0.9] },
      { direction: "lower" as const, values: [0.2, 0.3, 0.4] },
    ];
    expect(seedNote(rows, [false, true, true])).toBe("single-seed-mixed");
  });

  it("keeps the original wording when no group is compared", () => {
    expect(seedNote([{ direction: "higher", values: [0.9, 0.8] }], [false, false])).toBe(
      "single-seed",
    );
    expect(seedNote([{ direction: "higher", values: [0.9, 0.8] }])).toBe("single-seed");
  });
});
