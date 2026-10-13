import { describe, expect, it } from "vitest";

import {
  VIEW_H,
  VIEW_W,
  arcPath,
  chainMarks,
  chainPath,
  chainThrough,
  curve,
  curveAngle,
  curvePoint,
  makeColumn,
  neuronAt,
  sideArc,
  webLinks,
} from "./geometry";
import {
  bankLit,
  drawingFor,
  patchcoreDrawing,
  zoneOfPhase,
  type EpochFamily,
} from "./layouts";

describe("makeColumn", () => {
  it("stacks neurons around the middle of the box, top to bottom", () => {
    const col = makeColumn(100, 5);
    expect(col.ys).toHaveLength(5);
    expect([...col.ys].sort((a, b) => a - b)).toEqual(col.ys);
    // Centred: the middle neuron is the middle of the usable height.
    expect(col.ys[2]).toBeCloseTo((26 + 164) / 2);
  });

  it("puts a lone neuron in the middle", () => {
    expect(makeColumn(10, 1).ys).toEqual([95]);
  });

  it("keeps a crowded column inside the box", () => {
    const col = makeColumn(10, 9);
    expect(Math.min(...col.ys)).toBeGreaterThanOrEqual(26);
    expect(Math.max(...col.ys)).toBeLessThanOrEqual(164);
  });

  it("caps the gap, so a column of two stays together", () => {
    const col = makeColumn(10, 2, 6, 30);
    expect(col.ys[1] - col.ys[0]).toBe(30);
  });
});

describe("curves", () => {
  const a = { x: 10, y: 20 };
  const b = { x: 110, y: 100 };

  it("start and end where they are asked to", () => {
    expect(curve(a, b)).toBe("M10 20C60 20 60 100 110 100");
    const first = curvePoint(a, b, 0);
    const last = curvePoint(a, b, 1);
    expect(first).toEqual(a);
    expect(last.x).toBeCloseTo(b.x);
    expect(last.y).toBeCloseTo(b.y);
  });

  it("leave horizontally and descend through the middle", () => {
    expect(curveAngle(a, b, 0)).toBeCloseTo(0);
    expect(curveAngle(a, b, 1)).toBeCloseTo(0);
    expect(curveAngle(a, b, 0.5)).toBeGreaterThan(30);
    // Going back up reads as the opposite heading.
    expect(curveAngle(b, a, 0.5)).toBeLessThan(-100);
  });

  it("chain through several points with one segment each", () => {
    const d = chainPath([a, b, { x: 200, y: 60 }]);
    expect(d.startsWith("M10 20")).toBe(true);
    expect(d.match(/C/g)).toHaveLength(2);
    expect(chainPath([a])).toBe("");
  });

  it("put a mark in the middle of each segment of a chain", () => {
    const marks = chainMarks([a, b, { x: 200, y: 60 }]);
    expect(marks).toHaveLength(2);
    expect(marks[0].x).toBeCloseTo(60);
    expect(marks[0].y).toBeCloseTo(60);
  });

  it("arch over, or under, two neurons", () => {
    expect(arcPath({ x: 0, y: 50 }, { x: 100, y: 50 }, 30)).toBe("M0 50Q50 20 100 50");
    expect(arcPath({ x: 0, y: 50 }, { x: 100, y: 50 }, -30)).toBe("M0 50Q50 80 100 50");
    expect(sideArc(40, 10, 50, 16)).toBe("M40 10Q56 30 40 50");
  });
});

describe("webLinks", () => {
  it("leaves no neuron of either column out", () => {
    for (const [n, m] of [
      [5, 6],
      [6, 5],
      [3, 1],
      [1, 3],
      [2, 2],
      [6, 2],
    ]) {
      const links = webLinks(makeColumn(0, n), makeColumn(100, m));
      expect(new Set(links.map(([i]) => i)).size, `${n}->${m} sources`).toBe(n);
      expect(new Set(links.map(([, j]) => j)).size, `${n}->${m} targets`).toBe(m);
    }
  });

  it("is a tangle, not a full grid", () => {
    const links = webLinks(makeColumn(0, 6), makeColumn(100, 6));
    expect(links.length).toBeLessThan(36);
    expect(links.length).toBeGreaterThan(6);
  });
});

describe("chainThrough", () => {
  it("takes one neuron of each column, after the prefix", () => {
    const cols = [makeColumn(10, 3), makeColumn(50, 5)];
    const pts = chainThrough(cols, 0, [{ x: 0, y: 95 }]);
    expect(pts).toHaveLength(3);
    expect(pts[1]).toEqual({ x: 10, y: cols[0].ys[0] });
    expect(neuronAt(cols[1], 1).y).toBe(cols[1].ys[4]);
  });
});

const FAMILIES: EpochFamily[] = ["cnn", "transformer", "encoder_decoder", "detector", "autoencoder"];

describe("the layout of each family", () => {
  it("keeps every neuron, arc anchor and box inside the box", () => {
    for (const family of FAMILIES) {
      const d = drawingFor(family, "classes");
      for (const col of d.columns) {
        expect(col.x - col.r, family).toBeGreaterThanOrEqual(0);
        expect(col.x + col.r, family).toBeLessThanOrEqual(VIEW_W);
        for (const y of col.ys) {
          expect(y - col.r, family).toBeGreaterThanOrEqual(0);
          expect(y + col.r, family).toBeLessThanOrEqual(VIEW_H);
        }
      }
      for (const b of d.boxes) {
        expect(b.x, family).toBeGreaterThanOrEqual(0);
        expect(b.x + b.w, family).toBeLessThanOrEqual(VIEW_W);
        expect(b.y, family).toBeGreaterThanOrEqual(0);
        expect(b.y + b.h, family).toBeLessThanOrEqual(VIEW_H);
      }
    }
  });

  it("lays the columns out from left to right", () => {
    for (const family of FAMILIES) {
      const xs = drawingFor(family, "classes").columns.map((c) => c.x);
      expect([...xs].sort((a, b) => a - b), family).toEqual(xs);
    }
  });

  it("sends the forward pass from the data to the output, always rightwards", () => {
    for (const family of FAMILIES) {
      const d = drawingFor(family, "classes");
      expect(d.forward.length, family).toBe(3);
      for (const route of [...d.forward, d.validation]) {
        expect(route[0], family).toEqual(d.dataSource);
        for (let i = 1; i < route.length; i += 1) {
          expect(route[i].x, family).toBeGreaterThan(route[i - 1].x);
        }
      }
    }
  });

  it("writes the gradient's routes in forward order, two of them", () => {
    for (const family of FAMILIES) {
      const d = drawingFor(family, "classes");
      expect(d.backward.length, family).toBe(2);
      for (const route of d.backward) {
        for (let i = 1; i < route.length; i += 1) {
          expect(route[i].x, family).toBeGreaterThan(route[i - 1].x);
        }
      }
    }
  });

  it("webs every pair of neighbouring columns", () => {
    for (const family of FAMILIES) {
      const d = drawingFor(family, "classes");
      expect(d.webs, family).toHaveLength(d.columns.length - 1);
    }
  });

  it("ends a regression's network in a single neuron, and a classifier's in class scores", () => {
    for (const family of ["cnn", "transformer"] as const) {
      const last = (output: "classes" | "value") => {
        const cols = drawingFor(family, output).columns;
        return cols[cols.length - 1].ys.length;
      };
      expect(last("value"), family).toBe(1);
      expect(last("classes"), family).toBeGreaterThan(1);
    }
  });

  it("gives each family what makes it that family", () => {
    const by = Object.fromEntries(FAMILIES.map((f) => [f, drawingFor(f, "classes")]));
    // The U-Net has skip arcs, and the particles follow two of them.
    expect(by.encoder_decoder.arcs.filter((a) => a.kind === "skip")).toHaveLength(3);
    expect(by.encoder_decoder.skips).toHaveLength(2);
    // The U-Net and the autoencoder are symmetric around a bottleneck of two.
    for (const f of ["encoder_decoder", "autoencoder"] as const) {
      const sizes = by[f].columns.map((c) => c.ys.length);
      expect(sizes, f).toEqual([...sizes].reverse());
      expect(Math.min(...sizes), f).toBe(2);
    }
    // The autoencoder compares its output with its input; the U-Net has no such arc.
    expect(by.autoencoder.arcs.map((a) => a.kind)).toEqual(["recon"]);
    expect(by.autoencoder.skips).toEqual([]);
    // The transformer's tokens attend to one another, and it ends in a class token.
    expect(by.transformer.arcs.length).toBeGreaterThan(0);
    expect(by.transformer.arcs.every((a) => a.kind === "attention")).toBe(true);
    // The detector ends in three boxes of growing size, fed by three heads.
    expect(by.detector.boxes).toHaveLength(3);
    const areas = by.detector.boxes.map((b) => b.w * b.h);
    expect([...areas].sort((a, b) => a - b)).toEqual(areas);
    expect(by.detector.columns[by.detector.columns.length - 1].ys).toHaveLength(3);
    // The CNN narrows towards its head.
    const cnn = by.cnn.columns.map((c) => c.ys.length);
    expect(cnn[cnn.length - 1]).toBeLessThan(cnn[1]);
  });

  it("keeps the number of particles small", () => {
    // Three forward, two gradient, one validation, and at most two skips.
    for (const family of FAMILIES) {
      const d = drawingFor(family, "classes");
      expect(d.forward.length + d.backward.length + 1 + d.skips.length, family).toBeLessThanOrEqual(8);
    }
  });
});

describe("the PatchCore drawing", () => {
  const d = patchcoreDrawing();

  it("has a memory bank of thirty dots inside the box, filling left to right", () => {
    expect(d.bank).toHaveLength(30);
    for (const p of d.bank) {
      expect(p.x).toBeGreaterThan(0);
      expect(p.x).toBeLessThan(VIEW_W);
      expect(p.y).toBeGreaterThan(0);
      expect(p.y).toBeLessThan(VIEW_H);
    }
    const xs = d.bank.map((p) => p.x);
    expect([...xs].sort((a, b) => a - b)).toEqual(xs);
  });

  it("has one zone per phase, in the order of the phases", () => {
    expect(d.zones).toHaveLength(3);
    expect(d.zones.map((z) => z.x)).toEqual([...d.zones.map((z) => z.x)].sort((a, b) => a - b));
    expect(zoneOfPhase("extractingFeatures")).toBe(0);
    expect(zoneOfPhase("buildingBank")).toBe(1);
    expect(zoneOfPhase("scoring")).toBe(2);
    expect(zoneOfPhase(null)).toBe(-1);
  });

  it("ends the scoring route in the score", () => {
    for (const route of d.scoring) expect(route[route.length - 1]).toEqual(d.score);
  });
});

describe("bankLit", () => {
  const phase = (key: "extractingFeatures" | "buildingBank" | "scoring", fraction: number) => ({
    key,
    label: key,
    done: 0,
    total: 0,
    fraction,
  });

  it("stores nothing until the bank is being built", () => {
    expect(bankLit("preparing", null)).toBe(0);
    expect(bankLit("training", phase("extractingFeatures", 0.9))).toBe(0);
  });

  it("fills with the progress of its own phase", () => {
    expect(bankLit("training", phase("buildingBank", 0.5))).toBe(15);
    expect(bankLit("training", phase("buildingBank", 1))).toBe(30);
    // A run a stop cut keeps what it had stored.
    expect(bankLit("interrupted", phase("buildingBank", 0.2))).toBe(6);
  });

  it("is full from the scoring on, and when the run is done", () => {
    expect(bankLit("training", phase("scoring", 0.1))).toBe(30);
    expect(bankLit("testing", phase("scoring", 1))).toBe(30);
    expect(bankLit("done", phase("scoring", 1))).toBe(30);
  });
});
