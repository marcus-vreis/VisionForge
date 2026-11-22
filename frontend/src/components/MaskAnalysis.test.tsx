import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { coverageLabel, sharePct as pct, type MaskReport } from "../lib/mask-analysis";
import { MaskReportView } from "./MaskAnalysis";

function render(lang: "pt" | "en", report: MaskReport) {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={{ lang, t: lang === "pt" ? pt : en, locale: lang, setLang: () => {} }}>
      <MaskReportView report={report} />
    </LanguageContext.Provider>,
  );
}

const split = { present: true, images: 1, masks: 1, no_mask: [], no_image: [], no_mask_count: 0, no_image_count: 0 };

const clean: MaskReport = {
  task: "segmentation",
  num_classes: 2,
  ignore_index: 255,
  total_pairs: 2,
  scanned: 2,
  limit: 5000,
  sampled: false,
  cancelled: false,
  splits: { train: split, val: split, test: { ...split, present: false } },
  pairs: 2,
  classes: [
    { id: 0, ignore: false, out_of_range: false, pixels: 150, share: 75, shares: { train: 70, val: 80 } },
    { id: 1, ignore: false, out_of_range: false, pixels: 50, share: 25, shares: { train: 30, val: 20 } },
  ],
  absent_classes: [],
  coverage_bins: [
    { lo: 0, hi: 0, count: 0 },
    { lo: 0.25, hi: 0.5, count: 2 },
  ],
  empty_masks: 0,
  out_of_range_pixels: 0,
  out_of_range_ids: [],
  size_mismatch: 0,
  color_masks: 0,
  unreadable: 0,
  tiny_share: 0.0001,
  tiny_regions: [],
  tiny_images: 0,
  examples: {},
};

describe("the mask report (ADR-234)", () => {
  it.each(["pt", "en"] as const)("shows the class shares per split and no problem in %s", (lang) => {
    const t = (lang === "pt" ? pt : en).maskAnalysis;
    const html = render(lang, clean);
    expect(html).toContain(t.complete(2));
    expect(html).toContain(t.noProblems);
    expect(html).toContain("70.0%");
    expect(html).toContain("20.0%");
    // The test split is absent: no column for it.
    expect(html).not.toContain(`>${t.splitNames.test}<`);
    expect(html).toContain(t.noTiny(0.01));
  });

  it("names the ids outside the classes, color masks, missing partners and tiny regions", () => {
    const t = pt.maskAnalysis;
    const html = render("pt", {
      ...clean,
      splits: { ...clean.splits, train: { ...split, no_mask: ["x"], no_mask_count: 1 } },
      classes: [...clean.classes, { id: 9, ignore: false, out_of_range: true, pixels: 3, share: 1.5, shares: { train: 3 } }],
      out_of_range_pixels: 3,
      out_of_range_ids: [9],
      color_masks: 1,
      absent_classes: [],
      tiny_images: 1,
      tiny_regions: [{ id: 1, regions: 2 }],
      examples: { out_of_range: [{ split: "train", path: "train/masks/a.png" }] },
    });
    expect(html).toContain(t.outOfRange("9", 3, 1));
    expect(html).toContain(t.color(1));
    expect(html).toContain(t.unpaired(t.splitNames.train, 1, 0));
    expect(html).toContain(t.badId(9));
    expect(html).toContain(t.tiny(1, t.tinyClass(1, 2), 0.01));
    expect(html).toContain("train/masks/a.png");
    expect(html).not.toContain(t.noProblems);
  });

  it("says when the result is a sample", () => {
    const html = render("en", { ...clean, sampled: true, scanned: 2, total_pairs: 40 });
    expect(html).toContain(en.maskAnalysis.sampled(2, 40));
  });
});

describe("pixel shares", () => {
  it("keeps a rare class visible and zero as zero", () => {
    expect(pct(0)).toBe("0%");
    expect(pct(0.004)).toBe("<0.01%");
    expect(pct(0.36)).toBe("0.36%");
    expect(pct(93.24)).toBe("93.2%");
  });
});

describe("coverage labels", () => {
  it("names the empty bin and gives the others in percent", () => {
    expect(coverageLabel({ lo: 0, hi: 0, count: 1 }, "vazia")).toBe("vazia");
    expect(coverageLabel({ lo: 0, hi: 0.01, count: 1 }, "vazia")).toBe("0–1%");
    expect(coverageLabel({ lo: 0.75, hi: 1, count: 1 }, "vazia")).toBe("75–100%");
  });
});
