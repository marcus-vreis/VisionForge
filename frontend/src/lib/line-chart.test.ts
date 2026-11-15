import { describe, expect, it } from "vitest";

import {
  buildAreaPath,
  buildPath,
  isolatedPoints,
  nearestIndex,
  niceDomain,
  placeTooltip,
  runsOf,
  scaleLinear,
  uniqueXs,
  valueExtent,
  xTicks,
} from "./line-chart";

describe("niceDomain", () => {
  it("gives 3 to 5 round ticks that cover the data", () => {
    const d = niceDomain(0.31, 0.9)!;
    expect(d.ticks.length).toBeGreaterThanOrEqual(3);
    expect(d.ticks.length).toBeLessThanOrEqual(5);
    expect(d.min).toBeLessThanOrEqual(0.31);
    expect(d.max).toBeGreaterThanOrEqual(0.9);
    expect(d.ticks).toEqual([0.2, 0.4, 0.6, 0.8, 1]);
  });

  it("has no float noise in its ticks", () => {
    for (const tick of niceDomain(0.1, 0.7)!.ticks) {
      expect(String(tick).length).toBeLessThanOrEqual(5);
    }
  });

  it("gives a series that never moved one tick at its value", () => {
    const d = niceDomain(1, 1)!;
    expect(d.ticks).toEqual([1]);
    expect(d.min).toBeLessThan(1);
    expect(d.max).toBeGreaterThan(1);
    expect(niceDomain(0, 0)!.ticks).toEqual([0]);
  });

  it("resolves very small values", () => {
    const d = niceDomain(2.73e-6, 9.1e-6)!;
    expect(d.min).toBeLessThanOrEqual(2.73e-6);
    expect(d.max).toBeGreaterThanOrEqual(9.1e-6);
    expect(d.ticks.length).toBeGreaterThanOrEqual(3);
    expect(d.ticks.every((t) => t > 0)).toBe(true);
  });

  it("resolves a range tiny next to the values", () => {
    const d = niceDomain(1.0000001, 1.0000004)!;
    expect(d.ticks.length).toBeGreaterThanOrEqual(2);
    expect(d.ticks.length).toBeLessThanOrEqual(5);
  });

  it("accepts the bounds in either order and rejects non-finite ones", () => {
    expect(niceDomain(5, 1)).toEqual(niceDomain(1, 5));
    expect(niceDomain(Number.NaN, 1)).toBeNull();
    expect(niceDomain(0, Infinity)).toBeNull();
  });
});

describe("valueExtent and uniqueXs", () => {
  it("ignores gaps", () => {
    const a = [
      { x: 1, y: 3 },
      { x: 2, y: null },
      { x: 3, y: Number.NaN },
      { x: 4, y: -1 },
    ];
    expect(valueExtent([a])).toEqual({ min: -1, max: 3 });
    expect(
      uniqueXs([
        a,
        [
          { x: 2, y: 1 },
          { x: 4, y: 2 },
        ],
      ]),
    ).toEqual([1, 2, 4]);
  });

  it("is null when nothing was measured", () => {
    expect(valueExtent([[{ x: 1, y: null }], []])).toBeNull();
    expect(uniqueXs([[{ x: 1, y: undefined }]])).toEqual([]);
  });
});

describe("xTicks", () => {
  it("is the single epoch for one point", () => {
    expect(xTicks(3, 3)).toEqual([3]);
  });

  it("keeps the first and the last", () => {
    expect(xTicks(1, 3)).toEqual([1, 2, 3]);
    const t = xTicks(1, 10);
    expect(t[0]).toBe(1);
    expect(t.at(-1)).toBe(10);
  });

  it("stays within the count and uses integers for many epochs", () => {
    for (const hi of [37, 100, 300, 1000]) {
      const t = xTicks(1, hi, 6);
      expect(t.length).toBeLessThanOrEqual(7);
      expect(t.every(Number.isInteger)).toBe(true);
      expect(t[0]).toBe(1);
      expect(t.at(-1)).toBe(hi);
      expect([...t].sort((a, b) => a - b)).toEqual(t);
      expect(new Set(t).size).toBe(t.length);
    }
  });

  it("rounds fractional bounds and rejects NaN", () => {
    expect(xTicks(0.4, 2.6).every(Number.isInteger)).toBe(true);
    expect(xTicks(Number.NaN, 3)).toEqual([]);
  });
});

describe("nearestIndex", () => {
  const pos = [10, 20, 30, 40];

  it("snaps to the nearest position", () => {
    expect(nearestIndex(pos, 11)).toBe(0);
    expect(nearestIndex(pos, 26)).toBe(2);
    expect(nearestIndex(pos, 24)).toBe(1);
  });

  it("clamps outside the plot", () => {
    expect(nearestIndex(pos, -50)).toBe(0);
    expect(nearestIndex(pos, 500)).toBe(3);
  });

  it("sends a tie to the earlier epoch", () => {
    expect(nearestIndex(pos, 25)).toBe(1);
  });

  it("handles one point, none and a bad pointer", () => {
    expect(nearestIndex([7], 100)).toBe(0);
    expect(nearestIndex([], 5)).toBe(-1);
    expect(nearestIndex(pos, Number.NaN)).toBe(-1);
    expect(nearestIndex([5, 9], 6)).toBe(0);
    expect(nearestIndex([5, 9], 8)).toBe(1);
  });

  it("agrees with a linear scan over many epochs", () => {
    const many = Array.from({ length: 500 }, (_, i) => 50 + i * 1.3);
    for (const px of [0, 49, 50.6, 333.3, 700.1, 699, 1000]) {
      let best = 0;
      many.forEach((p, i) => {
        if (Math.abs(p - px) < Math.abs(many[best] - px)) best = i;
      });
      expect(nearestIndex(many, px)).toBe(best);
    }
  });
});

describe("scaleLinear", () => {
  it("maps and inverts", () => {
    expect(scaleLinear(5, 0, 10, 100, 200)).toBe(150);
    expect(scaleLinear(0.5, 0, 1, 100, 0)).toBe(50);
  });
  it("uses the middle of a degenerate domain", () => {
    expect(scaleLinear(3, 3, 3, 0, 10)).toBe(5);
  });
});

describe("placeTooltip", () => {
  const base = { anchorY: 50, width: 100, height: 60, chartWidth: 400, chartHeight: 150 };

  it("sits right of the guide line when it fits", () => {
    const p = placeTooltip({ ...base, anchorX: 100 });
    expect(p.side).toBe("right");
    expect(p.left).toBe(112);
  });

  it("flips to the left near the right edge", () => {
    const p = placeTooltip({ ...base, anchorX: 350 });
    expect(p.side).toBe("left");
    expect(p.left).toBe(238);
    expect(p.left + base.width).toBeLessThanOrEqual(base.chartWidth);
  });

  it("stays inside the chart on both axes", () => {
    const edge = placeTooltip({ ...base, anchorX: 5, anchorY: 2 });
    expect(edge.top).toBe(0);
    const low = placeTooltip({ ...base, anchorX: 5, anchorY: 149 });
    expect(low.top + base.height).toBeLessThanOrEqual(base.chartHeight);
    const narrow = placeTooltip({ ...base, anchorX: 20, chartWidth: 90 });
    expect(narrow.left).toBeGreaterThanOrEqual(0);
  });

  it("does not jump between neighbouring epochs on the same side", () => {
    const a = placeTooltip({ ...base, anchorX: 100 });
    const b = placeTooltip({ ...base, anchorX: 110 });
    expect(a.side).toBe(b.side);
    expect(b.left - a.left).toBe(10);
  });
});

describe("paths", () => {
  it("joins measured points with straight segments", () => {
    expect(
      buildPath([
        { x: 0, y: 1 },
        { x: 10, y: 2.345 },
        { x: 20, y: 3 },
      ]),
    ).toBe("M0 1 L10 2.35 L20 3");
  });

  it("breaks the line at a gap instead of bridging it", () => {
    const pts = [
      { x: 0, y: 1 },
      { x: 1, y: 2 },
      { x: 2, y: null },
      { x: 3, y: 5 },
      { x: 4, y: 6 },
    ];
    expect(buildPath(pts)).toBe("M0 1 L1 2 M3 5 L4 6");
    expect(runsOf(pts)).toHaveLength(2);
  });

  it("treats NaN like a gap", () => {
    expect(
      buildPath([
        { x: 0, y: 1 },
        { x: 1, y: Number.NaN },
        { x: 2, y: 3 },
      ]),
    ).toBe("");
  });

  it("leaves a lone point to be drawn as a dot", () => {
    const one = [{ x: 5, y: 7 }];
    expect(buildPath(one)).toBe("");
    expect(isolatedPoints(one)).toEqual(one);
    expect(
      isolatedPoints([
        { x: 0, y: 1 },
        { x: 1, y: 2 },
      ]),
    ).toEqual([]);
    expect(
      isolatedPoints([
        { x: 0, y: 1 },
        { x: 1, y: null },
        { x: 2, y: 3 },
      ]),
    ).toHaveLength(2);
  });

  it("has no path for no points", () => {
    expect(buildPath([])).toBe("");
    expect(buildAreaPath([], 10)).toBe("");
  });

  it("closes the area down to the base line", () => {
    expect(
      buildAreaPath(
        [
          { x: 0, y: 1 },
          { x: 10, y: 2 },
        ],
        20,
      ),
    ).toBe("M0 1 L10 2 L10 20 L0 20 Z");
  });
});
