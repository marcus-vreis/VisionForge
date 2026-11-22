import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import type { BoxReport } from "../lib/dataset-explorer";
import { BoxStatsCard } from "./BoxStatsCard";

/** React escapes "<" in text. */
const esc = (s: string) => s.replace(/</g, "&lt;");

function render(lang: "pt" | "en", boxes: BoxReport) {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={{ lang, t: lang === "pt" ? pt : en, locale: lang, setLang: () => {} }}>
      <BoxStatsCard boxes={boxes} />
    </LanguageContext.Provider>,
  );
}

const grid = Array.from({ length: 10 }, () => Array.from({ length: 10 }, () => 0));
grid[5][5] = 2;

const clean: BoxReport = {
  images: 2,
  boxes: 3,
  mean_per_image: 1.5,
  max_per_image: 2,
  per_image_bins: [
    { lo: 0, hi: 1, count: 0 },
    { lo: 1, hi: 2, count: 1 },
    { lo: 2, hi: 4, count: 1 },
    { lo: 32, hi: null, count: 0 },
  ],
  no_label_file: 0,
  empty_label: 0,
  size_thresholds: [1024, 9216],
  sizes: { small: 1, medium: 0, large: 2 },
  class_sizes: [{ name: "cat", small: 1, medium: 0, large: 2 }],
  unsized: 0,
  aspect_bins: [{ lo: 0.8, hi: 1.25, count: 3 }],
  heatmap: grid,
  pairs: [{ a: "cat", b: "dog", images: 1 }],
  invalid_count: 0,
  invalid: {},
  invalid_examples: [],
};

describe("the box statistics card (ADR-233)", () => {
  it.each(["pt", "en"] as const)("states the totals and COCO's size limits in %s", (lang) => {
    const t = (lang === "pt" ? pt : en).datasetExplorer;
    const html = render(lang, clean);
    expect(html).toContain(t.boxesTitle);
    expect(html).toContain(t.boxTotals(3, 2, 1.5, 2));
    // 1024 = 32², 9216 = 96²: the labels give the side, as COCO states them.
    expect(html).toContain(esc(t.boxSmall(32)));
    expect(html).toContain(t.boxMedium(32, 96));
    expect(html).toContain(t.boxPair("cat", "dog", 1));
    // Boxes per image as whole counts: 0, 1, 2-3.
    expect(html).toContain('title="2–3"');
    expect(html).not.toContain(t.boxEmpty(0, 0, 0).slice(0, 12));
  });

  it("keeps small, medium, large in order, whatever the counts", () => {
    const html = render("en", clean);
    const at = (s: string) => html.indexOf(esc(s));
    expect(at(en.datasetExplorer.boxSmall(32))).toBeLessThan(at(en.datasetExplorer.boxMedium(32, 96)));
    expect(at(en.datasetExplorer.boxMedium(32, 96))).toBeLessThan(at(en.datasetExplorer.boxLarge(96)));
  });

  it("warns about images without boxes and lists the invalid rows", () => {
    const html = render("pt", {
      ...clean,
      no_label_file: 1,
      empty_label: 2,
      invalid_count: 2,
      invalid: { zero_area: 1, outside_image: 1 },
      invalid_examples: [
        { split: "val", path: "images/val/b.png", line: 3, reason: "zero_area" },
        { split: "val", path: "images/val/c.png", line: 1, reason: "outside_image" },
      ],
    });
    const t = pt.datasetExplorer;
    expect(html).toContain(t.boxEmpty(3, 1, 2));
    expect(html).toContain(`${t.boxProblems.zero_area} (1)`);
    expect(html).toContain("images/val/b.png");
    expect(html).toContain(t.boxLine(3));
  });

  it("draws the heatmap as one cell per grid square, the fullest one opaque", () => {
    const html = render("en", clean);
    // Only the heatmap cells carry an opacity.
    expect(html.match(/;opacity:/g)).toHaveLength(100);
    expect(html).toContain('title="2" style="background:var(--accent-vf);opacity:1"');
  });
});
