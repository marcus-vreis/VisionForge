import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import type { SplitBalance } from "../lib/dataset-explorer";
import { SplitBalanceCard } from "./SplitBalanceCard";

function render(lang: "pt" | "en", balance: SplitBalance) {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={{ lang, t: lang === "pt" ? pt : en, locale: lang, setLang: () => {} }}>
      <SplitBalanceCard balance={balance} />
    </LanguageContext.Provider>,
  );
}

const stratified: SplitBalance = {
  splits: ["train", "val"],
  classes: [
    { name: "cat", counts: { train: 80, val: 8 }, shares: { train: 80, val: 80 }, gap_pp: 0, missing_in: [] },
    { name: "dog", counts: { train: 20, val: 2 }, shares: { train: 20, val: 20 }, gap_pp: 0, missing_in: [] },
  ],
  max_gap_pp: 0,
  cramers_v: 0,
  imbalance: { train: 4, val: 4 },
  verdict: "stratified",
  thresholds_pp: [2, 5],
};

describe("the split balance card (ADR-232)", () => {
  it.each(["pt", "en"] as const)("says a stratified division is fine in %s", (lang) => {
    const t = (lang === "pt" ? pt : en).datasetLab;
    const html = render(lang, stratified);
    expect(html).toContain(t.balanceTitle);
    expect(html).toContain(t.balanceVerdict.stratified(0, 2));
    // No "largest gaps" line when there is nothing to fix.
    expect(html).not.toContain(t.balanceTop("").split(":")[0]);
  });

  it("names a missing class, the largest gaps and points at re-splitting", () => {
    const html = render("pt", {
      ...stratified,
      classes: [
        { name: "fox", counts: { train: 5, val: 0 }, shares: { train: 5, val: 0 }, gap_pp: 5, missing_in: ["val"] },
        ...stratified.classes,
      ],
      max_gap_pp: 5,
      verdict: "missing_class",
    });
    expect(html).toContain(pt.datasetLab.balanceMissing("fox", pt.datasetExplorer.splitNames.val));
    expect(html).toContain("fox (5.0)");
    expect(html).toContain("Refazer divisão");
  });

  it("shows nothing for a dataset with one split", () => {
    expect(render("pt", { ...stratified, splits: ["train"], verdict: "one_split" })).toBe("");
  });
});
