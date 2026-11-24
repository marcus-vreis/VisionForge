import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import type { QualityMetricReport, QualityReport } from "../lib/dataset-explorer";
import { ImageQualityCard } from "./ImageQualityCard";

function render(lang: "pt" | "en", quality: QualityReport) {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={{ lang, t: lang === "pt" ? pt : en, locale: lang, setLang: () => {} }}>
      <ImageQualityCard quality={quality} root="/data/ds" />
    </LanguageContext.Provider>,
  );
}

const metric = (median: number, fences: [number, number]): QualityMetricReport => ({
  min: 10,
  q1: median - 5,
  median,
  q3: median + 5,
  max: 135,
  fences,
  median_by_split: { train: median, val: median - 2 },
});

const quality = (dark: number): QualityReport => ({
  count: 10,
  tukey_k: 1.5,
  measure_side: 256,
  metrics: { brightness: metric(117.5, [80, 150]), contrast: metric(50, [44, 56]), sharpness: metric(147.4, [69.1, 400]) },
  outlier_count: { dark, bright: 0, flat: 0, blurry: 0 },
  outliers: {
    dark: dark > 0 ? [{ split: "val", path: "cat/dark.png", value: 10 }] : [],
    bright: [],
    flat: [],
    blurry: [],
  },
});

describe("the brightness, contrast and sharpness card (ADR-238)", () => {
  it.each(["pt", "en"] as const)("shows the dark outlier with its thumbnail in %s", (lang) => {
    const t = (lang === "pt" ? pt : en).imageQuality;
    const html = render(lang, quality(1));
    expect(html).toContain(`${t.kinds.dark} (1)`);
    expect(html).toContain("/api/explorer/thumb?root=%2Fdata%2Fds&amp;path=cat%2Fdark.png");
    expect(html).toContain("cat/dark.png · 10.0");
    // Brightness range from the fences; contrast and sharpness only flag the low side.
    expect(html).toContain("80–150");
    expect(html).toContain("≥ 44.0");
    expect(html).toContain(t.medianIn((lang === "pt" ? pt : en).datasetExplorer.splitNames.val));
  });

  it("orders the split columns train, validation, test", () => {
    const q = quality(0);
    for (const m of Object.values(q.metrics!)) m.median_by_split = { test: 1, train: 2, val: 3 };
    const html = render("en", q);
    const at = (s: string) => html.indexOf(en.imageQuality.medianIn(s));
    expect(at("Train")).toBeLessThan(at("Validation"));
    expect(at("Validation")).toBeLessThan(at("Test"));
  });

  it("names anomaly groups as the screen does, train first and defect kinds last", () => {
    const q = quality(0);
    for (const m of Object.values(q.metrics!)) m.median_by_split = { "defect:hole": 1, test_normal: 2, train: 3 };
    const html = renderToStaticMarkup(
      <LanguageContext.Provider value={{ lang: "en", t: en, locale: "en", setLang: () => {} }}>
        <ImageQualityCard quality={q} root="/data/ds" splitLabel={(g) => `<${g}>`} />
      </LanguageContext.Provider>,
    );
    const at = (g: string) => html.indexOf(en.imageQuality.medianIn(`&lt;${g}&gt;`));
    expect(at("train")).toBeGreaterThan(-1);
    expect(at("train")).toBeLessThan(at("test_normal"));
    expect(at("test_normal")).toBeLessThan(at("defect:hole"));
  });

  it("counts past the thumbnails shown", () => {
    const html = render("en", quality(15));
    expect(html).toContain(en.imageQuality.more(14));
  });

  it("says when nothing is out of range", () => {
    const html = render("pt", quality(0));
    expect(html).toContain(pt.imageQuality.none);
    expect(html).not.toContain("<img");
  });

  it("renders nothing for a report without measures", () => {
    expect(render("pt", { count: 0 })).toBe("");
  });
});
