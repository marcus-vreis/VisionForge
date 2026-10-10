import { describe, expect, it } from "vitest";

import { FLOATING_WIDTH, placeFloating } from "./placement";

const box = (o: Partial<DOMRect>): DOMRect =>
  ({ left: 0, top: 0, width: 0, height: 0, right: 0, bottom: 0, ...o }) as DOMRect;

describe("placeFloating", () => {
  it("goes to the bottom-left corner when there is no target", () => {
    const p = placeFloating(null, 260, { width: 1280, height: 800 });

    expect(p).toEqual({ left: 20, top: 800 - 260 - 20 });
  });

  it("sits to the right of a target that leaves room there", () => {
    const sheet = box({ left: 300, right: 700, top: 100, bottom: 500, width: 400, height: 400 });

    const p = placeFloating(sheet, 260, { width: 1600, height: 800 });

    expect(p.left).toBe(716);
    expect(p.top).toBe(100 + 200 - 130);
  });

  it("falls back to the left side when only the left has room", () => {
    const sheet = box({ left: 900, right: 1500, top: 100, bottom: 500, width: 600, height: 400 });

    const p = placeFloating(sheet, 260, { width: 1600, height: 800 });

    expect(p.left).toBe(900 - 16 - FLOATING_WIDTH);
  });

  it("takes the left corner when neither side is wide enough", () => {
    // The training sheet on a 1280px window leaves 280px on each side.
    const sheet = box({ left: 280, right: 1000, top: 100, bottom: 700, width: 720, height: 600 });

    const p = placeFloating(sheet, 260, { width: 1280, height: 800 });

    expect(p).toEqual({ left: 20, top: 800 - 260 - 20 });
  });

  it("clears the 720px training sheet on a 1440px window", () => {
    // 360..1080: the corner card spans 20..360 and ends where the sheet begins.
    const sheet = box({ left: 360, right: 1080, top: 190, bottom: 610, width: 720, height: 420 });

    const p = placeFloating(sheet, 260, { width: 1440, height: 900 });

    expect(p.left + FLOATING_WIDTH).toBeLessThanOrEqual(sheet.left);
  });

  it("stays on screen vertically", () => {
    const sheet = box({ left: 100, right: 300, top: 700, bottom: 790, width: 200, height: 90 });

    const p = placeFloating(sheet, 260, { width: 1600, height: 800 });

    expect(p.top).toBe(800 - 260 - 20);
  });
});
