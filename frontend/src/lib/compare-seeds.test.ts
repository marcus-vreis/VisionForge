import { describe, expect, it } from "vitest";

import { seedNote, singleSeedValues } from "./compare-seeds";

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
  it("takes the group runs out of the values the note is judged on", () => {
    expect(singleSeedValues([0.9, 0.8, 0.7], [false, true, false])).toEqual([0.9, null, 0.7]);
    expect(singleSeedValues([0.9, 0.8], [false, false])).toEqual([0.9, 0.8]);
  });

  it("is silent when every compared run is a group: each cell carries its own interval", () => {
    const values = singleSeedValues([0.9, 0.8], [true, true]);
    expect(seedNote([{ direction: "higher", values }], true)).toBeNull();
  });

  it("is silent for one single run beside a group: no gap between single runs", () => {
    const values = singleSeedValues([0.9, 0.8], [false, true]);
    expect(seedNote([{ direction: "higher", values }], true)).toBeNull();
  });

  it("speaks of the single runs only when a group is in the selection too", () => {
    const values = singleSeedValues([0.9, 0.8, 0.85], [false, false, true]);
    expect(seedNote([{ direction: "higher", values }], true)).toBe("single-seed-mixed");
  });

  it("keeps the original wording when no group is compared", () => {
    expect(seedNote([{ direction: "higher", values: [0.9, 0.8] }], false)).toBe("single-seed");
  });
});
