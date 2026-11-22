import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import type { AnomalyReport } from "../lib/anomaly-analysis";
import { AnomalyReportView } from "./AnomalyAnalysis";

function render(lang: "pt" | "en", report: AnomalyReport) {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={{ lang, t: lang === "pt" ? pt : en, locale: lang, setLang: () => {} }}>
      <AnomalyReportView report={report} />
    </LanguageContext.Provider>,
  );
}

const group = (name: string) => ({ group: name, sizes: [{ size: "64×64", count: 20 }], distinct_sizes: 1, modes: { RGB: 20 } });

const clean: AnomalyReport = {
  task: "anomaly",
  total_images: 60,
  scanned: 60,
  limit: 20000,
  sampled: false,
  cancelled: false,
  max_distance: 6,
  few_images: 10,
  train_present: true,
  test_present: true,
  defect_kinds: ["crack"],
  extra_train_folders: [],
  ground_truth: false,
  counts: { train: 20, test_normal: 20, "defect:crack": 20 },
  groups: [group("defect:crack"), group("test_normal"), group("train")],
  size_mismatch: [],
  mode_mismatch: [],
  train_like_defect_count: 0,
  train_like_defect: [],
  test_like_train_count: 0,
  test_like_train: [],
  missing_masks: null,
  unreadable_count: 0,
  unreadable: [],
};

describe("the anomaly report (ADR-236)", () => {
  it.each(["pt", "en"] as const)("states the totals and no problem in %s", (lang) => {
    const t = (lang === "pt" ? pt : en).anomalyAnalysis;
    const html = render(lang, clean);
    expect(html).toContain(t.totals(20, 20, 20, 1));
    expect(html).toContain(t.noProblems);
    expect(html).toContain(t.defect("crack"));
  });

  it("names every finding", () => {
    const t = pt.anomalyAnalysis;
    const html = render("pt", {
      ...clean,
      counts: { train: 20, test_normal: 20, "defect:crack": 3 },
      extra_train_folders: ["bad"],
      size_mismatch: ["defect:crack"],
      mode_mismatch: ["test_normal"],
      train_like_defect_count: 1,
      train_like_defect: [{ path: "train/good/n1.png", like: "test/crack/s1.png", distance: 2 }],
      test_like_train_count: 1,
      test_like_train: [{ path: "test/good/c.png", like: "train/good/n0.png", distance: 0 }],
      ground_truth: true,
      missing_masks: { count: 1, paths: ["test/crack/s2.png"] },
    });
    expect(html).toContain(t.extraTrain("bad"));
    expect(html).toContain(t.few("crack (3)", 10));
    expect(html).toContain(t.trainLikeDefect(1));
    expect(html).toContain(t.testLikeTrain(1));
    expect(html).toContain(t.sizeMismatch(t.defect("crack")));
    expect(html).toContain(t.modeMismatch(t.testNormal));
    expect(html).toContain(t.missingMasks(1));
    expect(html).toContain("train/good/n1.png ≈ test/crack/s1.png (2)");
    expect(html).not.toContain(t.noProblems);
  });

  it("says when the test has no defect folder", () => {
    const html = render("en", { ...clean, defect_kinds: [], counts: { train: 20, test_normal: 20 } });
    expect(html).toContain(en.anomalyAnalysis.noDefects);
  });
});
