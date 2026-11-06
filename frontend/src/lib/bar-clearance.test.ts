import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { BAR_GAP, barClearance } from "./bar-clearance";

describe("phone-width bar CSS", () => {
  const css = readFileSync(resolve(__dirname, "../index.css"), "utf-8");
  const phone = css.slice(css.indexOf("@media (max-width: 640px) {\n  .vf-app-main"));

  it("wraps the bar and gives Treinar its own full row", () => {
    expect(phone).toMatch(/\.vf-bottom-bar \{[^}]*flex-wrap: wrap !important/);
    expect(phone).toMatch(/\.vf-bottom-bar-train \{[^}]*flex: 1 1 100% !important/);
  });

  it("pads the page by the measured bar height", () => {
    expect(phone).toMatch(/padding-bottom: max\(96px, var\(--vf-bar-clearance, 0px\)\) !important/);
  });
});

describe("barClearance", () => {
  it("covers the bar, its offset from the edge and the gap", () => {
    // 812 px viewport, bar top at 700 (104 px bar, 8 px above the edge, ends 804).
    expect(barClearance(812, 700)).toBe(112 + BAR_GAP);
  });

  it("grows when the bar wraps and gets taller", () => {
    expect(barClearance(812, 650)).toBeGreaterThan(barClearance(812, 700));
  });

  it("rounds up fractional positions and never goes negative", () => {
    expect(barClearance(800, 700.4, 0)).toBe(100);
    expect(barClearance(800, 900)).toBe(0);
  });
});
