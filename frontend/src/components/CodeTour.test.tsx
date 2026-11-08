import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import { SHEET_TABS } from "../lib/training-story";
import { CodeTour } from "./CodeTour";

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };
const wrap = (element: React.ReactNode) =>
  renderToStaticMarkup(<LanguageContext.Provider value={lang}>{element}</LanguageContext.Provider>);

describe("CodeTour", () => {
  it("is the last tab of the training sheet", () => {
    expect(SHEET_TABS).toEqual(["log", "results", "story", "code"]);
  });
  it("says the annotated code exists for classification only", () => {
    const html = wrap(<CodeTour taskKey="detection" settings={null} accent="red" />);
    expect(html).toContain(pt.codeTour.onlyClassification);
  });
  it("loads the classification code from the server", () => {
    const html = wrap(<CodeTour taskKey="classification" settings={{ kind: "binary" }} accent="red" />);
    expect(html).toContain(pt.codeTour.loading);
  });
  it("words line ranges", () => {
    expect(pt.codeTour.lines(10, 10)).toBe("linha 10");
    expect(pt.codeTour.lines(10, 14)).toBe("linhas 10–14");
  });
});
