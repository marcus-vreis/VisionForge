import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { num, type TargetReport, type TargetSplit } from "../lib/target-analysis";
import { TargetReportView } from "./TargetAnalysis";

function render(lang: "pt" | "en", report: TargetReport) {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={{ lang, t: lang === "pt" ? pt : en, locale: lang, setLang: () => {} }}>
      <TargetReportView report={report} />
    </LanguageContext.Provider>,
  );
}

const split = (over: Partial<TargetSplit> = {}): TargetSplit => ({
  count: 4,
  bad: 0,
  mean: 2.5,
  std: 1.29,
  min: 1,
  q1: 1.75,
  median: 2.5,
  q3: 3.25,
  max: 4,
  histogram: [1, 1, 1, 1],
  outlier_count: 0,
  outliers: [],
  shift_sd: 0,
  ...over,
});

const csv = { present: true, rows: 4, missing_columns: [], missing_images: 0, missing_examples: [], bad_examples: [] };

const clean: TargetReport = {
  task: "regression",
  cancelled: false,
  tukey_k: 1.5,
  splits: { train: csv, val: csv, test: { ...csv, present: false, rows: 0 } },
  targets: [{ name: "age", range: [1, 4], bins: 4, fences: [-0.5, 5.5], splits: { train: split(), val: split({ shift_sd: 0.1 }) } }],
  correlations: [],
};

describe("the target report (ADR-235)", () => {
  it.each(["pt", "en"] as const)("shows the summary per split and no problem in %s", (lang) => {
    const t = (lang === "pt" ? pt : en).targetAnalysis;
    const html = render(lang, clean);
    expect(html).toContain(t.noProblems);
    expect(html).toContain(">age<");
    expect(html).toContain(">2.5<");
    // One line per split that has values; the absent test split draws none.
    expect(html.match(/<polyline/g)).toHaveLength(2);
    // A shift of 0.1 sd is under the 0.5 sd threshold: no warning.
    const val = (lang === "pt" ? pt : en).analysisJob.splitNames.val;
    expect(html).not.toContain(t.shifted(val, "0.1"));
  });

  it("warns about a shifted split, missing images, bad cells and lists the outliers", () => {
    const t = pt.targetAnalysis;
    const html = render("pt", {
      ...clean,
      splits: {
        ...clean.splits,
        train: { ...csv, missing_images: 1, missing_examples: [{ line: 3, image: "x.png" }], bad_examples: [{ line: 5, target: "age", cell: "n/a" }] },
      },
      targets: [
        {
          ...clean.targets[0],
          splits: {
            train: split({ outlier_count: 1, outliers: [{ line: 4, image: "big.png", value: 99 }] }),
            val: split({ shift_sd: -0.8 }),
          },
        },
      ],
      correlations: [{ a: "age", b: "weight", r: 0.91234 }],
    });
    expect(html).toContain(t.shifted(pt.analysisJob.splitNames.val, "-0.8"));
    expect(html).toContain(t.missingImages(pt.analysisJob.splitNames.train, 1, 4));
    expect(html).toContain(t.badCells(1));
    expect(html).toContain(t.outliersList(1, "-0.5", "5.5"));
    expect(html).toContain("big.png");
    expect(html).toContain(t.correlation("age", "weight", "0.912"));
  });
});

describe("number formatting", () => {
  it("keeps four significant digits and switches to exponent at the extremes", () => {
    expect(num(null)).toBe("—");
    expect(num(0)).toBe("0");
    expect(num(2.345678)).toBe("2.346");
    expect(num(123456)).toBe("1.23e+5");
    expect(num(0.0001234)).toBe("1.23e-4");
  });
});
