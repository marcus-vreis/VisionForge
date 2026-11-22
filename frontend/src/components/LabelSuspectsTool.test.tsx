import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import type { LabelSuspectsReport, SuspectSplitReport } from "../lib/label-suspects";
import { LabelSuspectsReportView } from "./LabelSuspectsTool";

function render(lang: "pt" | "en", report: LabelSuspectsReport) {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={{ lang, t: lang === "pt" ? pt : en, locale: lang, setLang: () => {} }}>
      <LabelSuspectsReportView report={report} />
    </LanguageContext.Provider>,
  );
}

const split = (over: Partial<SuspectSplitReport> = {}): SuspectSplitReport => ({
  folder: "/data/val",
  images: 5,
  failed: 0,
  in_sample: false,
  thresholds: [0.57, 0.85],
  suspect_count: 1,
  suspects: [{ path: "cat/c3.png", given: "cat", suspected: "dog", p_given: 0.02, p_suspected: 0.98 }],
  joint: [
    [2, 1],
    [0, 2],
  ],
  pairs: [{ given: "cat", taken_for: "dog", images: 1 }],
  ...over,
});

const report: LabelSuspectsReport = {
  run_dir: "/runs/x",
  computed_at: "2026-11-22T10:00:00+00:00",
  class_names: ["cat", "dog"],
  splits: { val: split() },
  missing_splits: ["test"],
};

describe("the suspicious labels report (ADR-237)", () => {
  it.each(["pt", "en"] as const)("lists the suspect with both probabilities in %s", (lang) => {
    const t = (lang === "pt" ? pt : en).labelSuspects;
    const html = render(lang, report);
    expect(html).toContain(t.found(1, 5));
    expect(html).toContain("cat/c3.png");
    expect(html).toContain("2.0%");
    expect(html).toContain("98.0%");
    expect(html).toContain(t.missing(t.splitNames.test));
    expect(html).toContain(t.pair("cat", "dog", 1));
  });

  it("warns that training is in-sample and points at the CSV past the shown rows", () => {
    const many = Array.from({ length: 31 }, (_, i) => ({
      path: `cat/${i}.png`,
      given: "cat",
      suspected: "dog",
      p_given: 0.1,
      p_suspected: 0.9,
    }));
    const html = render("pt", {
      ...report,
      splits: { train: split({ in_sample: true, suspect_count: 31, suspects: many }) },
      missing_splits: [],
    });
    expect(html).toContain(pt.labelSuspects.inSample);
    expect(html).toContain(pt.labelSuspects.more(1, "train"));
    expect(html.match(/<tr>/g)).toHaveLength(31); // header + 30 shown
  });

  it("says when a split has no suspect", () => {
    const html = render("en", { ...report, splits: { val: split({ suspect_count: 0, suspects: [], pairs: [] }) } });
    expect(html).toContain(en.labelSuspects.found(0, 5));
    expect(html).not.toContain("<table");
  });
});
