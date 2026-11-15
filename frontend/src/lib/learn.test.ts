import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import {
  CONFUSION_EXAMPLE,
  CURVE_EPOCHS,
  LEARN_CHAPTERS,
  PROJECT_DOCS,
  RESOURCES,
  STACK,
  accuracyOf,
  bestEpoch,
  chapterAnchor,
  lossCurve,
  splitParts,
} from "./learn";

const DICTS = { pt, en };

describe("the chapters", () => {
  it("have a title and at least one paragraph in both languages", () => {
    for (const dict of Object.values(DICTS)) {
      expect(Object.keys(dict.learn.chapters)).toEqual([...LEARN_CHAPTERS]);
      for (const id of LEARN_CHAPTERS) {
        const chapter = dict.learn.chapters[id];
        expect(chapter.title.length).toBeGreaterThan(0);
        expect(chapter.body.length).toBeGreaterThan(0);
        for (const paragraph of chapter.body) expect(paragraph.length).toBeGreaterThan(40);
      }
    }
  });

  it("have one anchor each", () => {
    const anchors = LEARN_CHAPTERS.map(chapterAnchor);
    expect(new Set(anchors).size).toBe(LEARN_CHAPTERS.length);
  });
});

describe("splitParts", () => {
  it("divides 40 images into 28, 6 and 6, in that order", () => {
    const parts = splitParts(40);
    expect(parts.filter((p) => p === "train")).toHaveLength(28);
    expect(parts.filter((p) => p === "val")).toHaveLength(6);
    expect(parts.filter((p) => p === "test")).toHaveLength(6);
    expect(parts.join(",")).toMatch(/^(train,)+(val,)+(test,?)+$/);
  });

  it("gives every image a part", () => {
    for (const count of [0, 1, 7, 100]) expect(splitParts(count)).toHaveLength(count);
  });
});

describe("lossCurve", () => {
  it("draws one point per epoch, numbered from 1", () => {
    for (const choice of ["low", "good", "high"] as const) {
      const points = lossCurve(choice);
      expect(points.map((p) => p.epoch)).toEqual(
        Array.from({ length: CURVE_EPOCHS }, (_, i) => i + 1),
      );
      for (const p of points) {
        expect(Number.isFinite(p.train)).toBe(true);
        expect(Number.isFinite(p.val)).toBe(true);
      }
    }
  });

  it("is the same on every call", () => {
    expect(lossCurve("high")).toEqual(lossCurve("high"));
  });

  it("at a low rate is still falling at the last epoch", () => {
    const points = lossCurve("low");
    const last = points[points.length - 1];
    expect(last.val).toBeLessThan(points[19].val);
    expect(points[19].val).toBeLessThan(points[9].val);
    // It never got as low as the right rate does.
    expect(last.train).toBeGreaterThan(0.5);
    expect(bestEpoch(points)).toBeGreaterThan(CURVE_EPOCHS - 3);
  });

  it("at the right rate overfits: validation rises after its best epoch while training keeps falling", () => {
    const points = lossCurve("good");
    const best = bestEpoch(points);
    const atBest = points[best - 1];
    const last = points[points.length - 1];
    expect(best).toBeGreaterThan(5);
    expect(best).toBeLessThan(CURVE_EPOCHS - 5);
    expect(last.val).toBeGreaterThan(atBest.val + 0.1);
    expect(last.train).toBeLessThan(atBest.train);
    expect(atBest.val).toBeLessThan(0.4);
  });

  it("at a high rate never settles", () => {
    const tail = lossCurve("high")
      .slice(-10)
      .map((p) => p.train);
    expect(Math.max(...tail) - Math.min(...tail)).toBeGreaterThan(0.2);
    expect(Math.min(...tail)).toBeGreaterThan(0.5);
  });
});

describe("bestEpoch", () => {
  it("is the epoch with the lowest validation loss", () => {
    const points = [
      { epoch: 1, train: 1, val: 0.9 },
      { epoch: 2, train: 0.5, val: 0.4 },
      { epoch: 3, train: 0.2, val: 0.6 },
    ];
    expect(bestEpoch(points)).toBe(2);
  });
});

describe("accuracyOf", () => {
  it("is the diagonal over the total", () => {
    expect(accuracyOf(CONFUSION_EXAMPLE)).toBeCloseTo(51 / 60);
    expect(
      accuracyOf([
        [3, 1],
        [0, 4],
      ]),
    ).toBeCloseTo(7 / 8);
  });

  it("is 0 for an empty matrix", () => {
    expect(accuracyOf([])).toBe(0);
  });

  it("matches what the caption says about class B", () => {
    // "B foi confundida com C duas vezes e com A outras duas."
    expect(CONFUSION_EXAMPLE[1]).toEqual([2, 16, 2]);
  });
});

describe("the links", () => {
  const all = [...STACK, ...RESOURCES, ...PROJECT_DOCS];

  it("are https and each appears once", () => {
    for (const link of all) expect(link.url).toMatch(/^https:\/\/[^\s]+$/);
    expect(new Set(all.map((l) => l.url)).size).toBe(all.length);
    expect(new Set(all.map((l) => l.id)).size).toBe(all.length);
  });

  it("have a line in both languages", () => {
    for (const dict of Object.values(DICTS)) {
      const c = dict.learn.chapters;
      expect(Object.keys(c.stack.items).sort()).toEqual(STACK.map((s) => s.id).sort());
      expect(Object.keys(c.more.resources).sort()).toEqual(RESOURCES.map((r) => r.id).sort());
      expect(Object.keys(c.more.docs).sort()).toEqual(PROJECT_DOCS.map((d) => d.id).sort());
    }
  });

  it("to the project's pages name files that exist in this repository", () => {
    const marker = "/VisionForge/blob/main/";
    for (const doc of PROJECT_DOCS) {
      const at = doc.url.indexOf(marker);
      expect(at).toBeGreaterThan(0);
      const relative = doc.url.slice(at + marker.length);
      const file = fileURLToPath(new URL(`../../../${relative}`, import.meta.url));
      expect(existsSync(file), relative).toBe(true);
    }
  });
});
