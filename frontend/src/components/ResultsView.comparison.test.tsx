import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { comparisonGroupFromReport } from "../lib/replicated-comparison-report";
import type { RunResult } from "../types/run";
import { ResultsView } from "./ResultsView";

const aggregate = (mean: number) => ({
  n: 2,
  mean,
  std: 0.01,
  std_ddof: 1,
  min: mean - 0.01,
  max: mean + 0.01,
  ci95_low: mean - 0.09,
  ci95_high: mean + 0.09,
});

/** What `finish_replicated_comparison` hands the result route (ADR-061). */
const report = {
  kind: "replicated_comparison",
  metric: "r2",
  metric_direction: "higher",
  seeds: [42, 43],
  alpha: 0.05,
  variants: {
    baseline: {
      overrides: {},
      aggregates: { r2: aggregate(0.71) },
      trials: [
        { seed: 42, status: "success", metrics: { r2: 0.7 }, run_dir: "out/models/reg_baseline/run_a", error: "" },
        { seed: 43, status: "success", metrics: { r2: 0.72 }, run_dir: "out\\models\\reg_baseline\\run_b", error: "" },
      ],
      successful: 2,
      per_seed: { "42": 0.7, "43": 0.72 },
    },
    "learning_rate=0.01": {
      overrides: { "training.learning_rate": 0.01 },
      aggregates: { r2: aggregate(0.64) },
      trials: [
        { seed: 42, status: "success", metrics: { r2: 0.63 }, run_dir: "out/models/reg_lr/run_c", error: "" },
        { seed: 43, status: "failed", metrics: {}, run_dir: "", error: "boom" },
      ],
      successful: 1,
      per_seed: { "42": 0.63 },
    },
  },
  comparisons: [
    {
      label_a: "baseline",
      label_b: "learning_rate=0.01",
      metric: "r2",
      n_pairs: 2,
      mean_a: 0.71,
      mean_b: 0.64,
      mean_difference: 0.07,
      test: "wilcoxon",
      test_reason: "2 pairs: too few to check normality",
      p_value: 0.5,
      effect_size: 3.1,
      effect_label: "large",
      min_achievable_p: 0.5,
      underpowered: true,
      significant: false,
    },
  ],
  best_by_mean: "baseline",
  ranked_by_mean: ["baseline", "learning_rate=0.01"],
  ranking_seeds: [42],
  significant_pairs: 0,
  skipped_variants: [],
  underpowered: true,
  power_note: "With 2 paired seeds the rank test cannot return a p-value below 0.5000",
  not_run: [],
  stopped: false,
  report_dir: "out/reports/reg",
};

const result: RunResult = {
  run_id: "reg_20261111_120000_000000",
  metrics: {},
  report,
  artifacts: {},
};

function render(lang: "pt" | "en", onOpenInHistory?: (id: string) => void): string {
  const dict = lang === "pt" ? pt : en;
  return renderToStaticMarkup(
    <LanguageContext.Provider
      value={{ lang, t: dict, locale: lang === "pt" ? "pt-BR" : "en-US", setLang: () => {} }}
    >
      <ResultsView
        result={result}
        onClose={() => {}}
        taskAccent="#5ad"
        onOpenInHistory={onOpenInHistory}
      />
    </LanguageContext.Provider>,
  );
}

describe("ResultsView for a replicated comparison (ADR-193)", () => {
  it("does not dump the report as JSON", () => {
    const html = render("pt", () => {});

    expect(html).not.toContain(pt.resultsView.reportTitle);
    expect(html).not.toContain("<pre");
    expect(html).not.toContain("&quot;variants&quot;");
  });

  it.each(["pt", "en"] as const)("shows the variants and the paired matrix (%s)", (lang) => {
    const dict = lang === "pt" ? pt : en;
    const html = render(lang, () => {});

    expect(html).toContain(dict.runGroup.detail.variantsTitle);
    expect(html).toContain(dict.runGroup.detail.testsTitle);
    expect(html).toContain("learning_rate=0.01");
    expect(html).toContain("training.learning_rate=0.01");
    expect(html).toContain(dict.runGroup.detail.tests.wilcoxon);
    expect(html).toContain(dict.runGroup.detail.underpowered);
    expect(html).toContain(dict.resultsView.openGroupInHistory);
  });

  it("offers no History button when nothing can open it", () => {
    expect(render("pt")).not.toContain(pt.resultsView.openGroupInHistory);
  });
});

describe("comparisonGroupFromReport", () => {
  it("reads the report the way the History group copies it", () => {
    const group = comparisonGroupFromReport(report);

    expect(group.kind).toBe("replicated_comparison");
    expect(group.n_requested).toBe(4);
    expect(group.n_finished).toBe(3);
    expect(group.best_by_mean).toBe("baseline");
    expect(group.comparisons).toHaveLength(1);
    const baseline = group.variants?.["baseline"];
    expect(baseline?.seeds_finished).toEqual([42, 43]);
    // The seed's run id is its folder's name, on either separator.
    expect(baseline?.children.map((c) => c.run_id)).toEqual(["run_a", "run_b"]);
    const lr = group.variants?.["learning_rate=0.01"];
    expect(lr?.seeds_finished).toEqual([42]);
    expect(lr?.children[1].run_id).toBeNull();
  });

  it("counts the variants a stop kept from starting among the trainings asked for", () => {
    const group = comparisonGroupFromReport({ ...report, not_run: ["extra"] });

    expect(group.n_requested).toBe(6);
    expect(group.not_run).toEqual(["extra"]);
  });
});
