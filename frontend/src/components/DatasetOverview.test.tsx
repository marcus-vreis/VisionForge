import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import type { ScanReport } from "../lib/dataset-explorer";
import { Report, SavedNote } from "./DatasetOverview";

const clean: ScanReport = {
  total_images: 2, scanned: 2, limit: 20000, sampled: false, sample_step: 1,
  cancelled: false, complete: true, corrupted_count: 0, corrupted: [],
  duplicate_group_count: 0, duplicate_file_count: 0, duplicate_groups: [],
  leakage_group_count: 0, leakage_groups: [], sizes: { count: 0 },
};

function render(lang: "pt" | "en", node: React.ReactNode) {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={{ lang, t: lang === "pt" ? pt : en, locale: lang, setLang: () => {} }}>
      {node}
    </LanguageContext.Provider>,
  );
}

describe("the Laboratory scan report (ADR-231)", () => {
  it.each(["pt", "en"] as const)("lists near-duplicates across splits as a problem in %s", (lang) => {
    const t = (lang === "pt" ? pt : en).datasetExplorer;
    const report = {
      ...clean, near_checked: true, near_max_distance: 6, near_group_count: 1,
      near_groups: [{ splits: ["train", "val"], paths: ["train/cat/a.png", "val/cat/a.png"] }],
    };
    const html = render(lang, <Report report={report} root="/data" />);
    expect(html).toContain(`${t.nearTitle} (1)`);
    expect(html).toContain("val/cat/a.png");
    expect(html).not.toContain(t.noProblems);
    expect(html).not.toContain(t.nearNotChecked);
  });

  it("says near-duplicates were not looked for, also in a scan kept before the option", () => {
    const html = render("pt", <Report report={clean} root="/data" />);
    expect(html).toContain(pt.datasetExplorer.nearNotChecked);
    expect(html).toContain(pt.datasetExplorer.noProblems);
  });

  it("marks a kept scan stale when the dataset changed", () => {
    const fresh = render("pt", <SavedNote saved={{ report: clean, saved_at: "2026-11-21T18:00:00+00:00", stale: false }} />);
    const stale = render("pt", <SavedNote saved={{ report: clean, saved_at: "2026-11-21T18:00:00+00:00", stale: true }} />);
    expect(fresh).toContain("Última varredura guardada:");
    expect(fresh).not.toContain(pt.datasetExplorer.savedStale);
    expect(stale).toContain(pt.datasetExplorer.savedStale);
  });
});
