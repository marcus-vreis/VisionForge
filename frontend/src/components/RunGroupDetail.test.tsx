import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import type { GroupChild, RunGroup } from "../types/run";
import { RunGroupDetail } from "./RunGroupDetail";

const fold = (i: number, r2: number): GroupChild => ({
  seed: null,
  fold: i,
  variant: null,
  status: "success",
  run_id: `reg_fold${i}_20260101`,
  run_dir: `C:/models/reg_fold${i}/20260101`,
  metrics: { r2 },
  error: "",
});

const kfold: RunGroup = {
  kind: "kfold",
  metric: "r2",
  seeds: [],
  n_requested: 3,
  n_finished: 3,
  stopped: false,
  std_ddof: 1,
  report_path: null,
  report_dir: null,
  children: [fold(0, 0.81), fold(1, 0.79), fold(2, 0.83)],
  aggregates: {
    r2: { n: 3, mean: 0.81, std: 0.02, std_ddof: 1, min: null, max: null, ci95_low: null, ci95_high: null },
  },
};

function render(group: RunGroup, lang: "pt" | "en" = "en"): string {
  return renderToStaticMarkup(
    <LanguageContext.Provider
      value={{ lang, t: lang === "pt" ? pt : en, locale: lang === "pt" ? "pt-BR" : "en-US", setLang: () => {} }}
    >
      <RunGroupDetail
        group={group}
        accent="#0f0"
        knownRunIds={new Set((group.children ?? []).map((c) => c.run_id ?? ""))}
        onOpenRun={() => {}}
      />
    </LanguageContext.Provider>,
  );
}

describe("RunGroupDetail for a K-fold (ADR-123)", () => {
  it("lists folds, not seeds, with each fold's number", () => {
    const html = render(kfold);
    expect(html).toContain(en.runGroup.detail.eyebrow.kfold);
    expect(html).toContain(en.runGroup.detail.foldsTitle);
    expect(html).not.toContain(en.runGroup.detail.seedsAsked);
    expect(html).toContain("0.8100");
    for (const i of [0, 1, 2]) expect(html).toContain(`reg_fold${i}_20260101`);
  });

  it("does not warn about few seeds: three folds is the design", () => {
    const html = render(kfold);
    expect(html).not.toContain(en.runGroup.detail.fewSeeds(3));
  });

  it("speaks Portuguese", () => {
    expect(render(kfold, "pt")).toContain(pt.runGroup.detail.eyebrow.kfold);
  });
});
