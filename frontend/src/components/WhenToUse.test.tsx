import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { StrategyCard, type PanelStrategy } from "./ExperimentHeader";
import { WhenToUse } from "./WhenToUse";

const langs = {
  pt: { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} },
  en: { lang: "en" as const, t: en, locale: "en-US", setLang: () => {} },
};
const wrap = (element: React.ReactNode, lang: keyof typeof langs = "pt") =>
  renderToStaticMarkup(
    <LanguageContext.Provider value={langs[lang]}>{element}</LanguageContext.Provider>,
  );

const PANEL_STRATEGIES: PanelStrategy[] = [
  "simple",
  "cv",
  "sweep",
  "replicates",
  "compare",
  "replicatedComparison",
];

describe("WhenToUse", () => {
  it("says, in the language of the page, when the strategy is the right one", () => {
    expect(wrap(<WhenToUse strategy="cv" />)).toContain(pt.whenToUse.strategies.cv);
    expect(wrap(<WhenToUse strategy="cv" />, "en")).toContain(en.whenToUse.strategies.cv);
    expect(wrap(<WhenToUse strategy="cv" />, "en")).toContain("When to use");
  });

  it("keeps each sentence to one short line of advice", () => {
    for (const dict of [pt, en]) {
      for (const sentence of Object.values(dict.whenToUse.strategies)) {
        expect(sentence.length).toBeLessThanOrEqual(110);
        expect(sentence.trim().endsWith(".")).toBe(true);
      }
    }
  });

  it.each(PANEL_STRATEGIES)(
    "the strategy card shows the sentence of %s when it is selected",
    (strategy) => {
      const html = wrap(<StrategyCard strategy={strategy} onStrategyChange={() => {}} />);
      expect(html).toContain(`data-when-to-use="${strategy}"`);
      expect(html).toContain(pt.whenToUse.strategies[strategy]);
    },
  );
});
