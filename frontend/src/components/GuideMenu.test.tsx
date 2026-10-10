/**
 * The test setup has no DOM, so the menu is rendered to markup with its list
 * open (`initialOpen`) and the closed state is read from the same component.
 * Clicking is the one thing left to the browser check.
 */
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { GUIDES } from "../lib/guides";
import { GuideMenu } from "./GuideMenu";

const inLanguage = (lang: "pt" | "en", node: React.ReactNode) =>
  renderToStaticMarkup(
    <LanguageContext.Provider
      value={{
        lang,
        t: lang === "pt" ? pt : en,
        locale: lang === "pt" ? "pt-BR" : "en-US",
        setLang: () => {},
      }}
    >
      {node}
    </LanguageContext.Provider>,
  );

describe("GuideMenu", () => {
  it("shows only the button until it is opened", () => {
    const html = inLanguage("pt", <GuideMenu onSelect={() => {}} />);

    expect(html).toContain(pt.header.guide);
    expect(html).toContain('aria-expanded="false"');
    expect(html).not.toContain('role="menu"');
  });

  it("lists every guide of the registry when open, in order", () => {
    for (const lang of ["pt", "en"] as const) {
      const dict = lang === "pt" ? pt : en;
      const html = inLanguage(lang, <GuideMenu onSelect={() => {}} initialOpen />);

      expect(html).toContain('aria-expanded="true"');
      expect(html).toContain('role="menu"');
      const positions = GUIDES.map((g) => html.indexOf(g.title(dict)));
      expect(positions.every((p) => p >= 0)).toBe(true);
      expect([...positions].sort((a, b) => a - b)).toEqual(positions);
      expect(html.match(/role="menuitem"/g)).toHaveLength(GUIDES.length);
    }
  });

  it("names the two guides in Portuguese", () => {
    const html = inLanguage("pt", <GuideMenu onSelect={() => {}} initialOpen />);

    expect(html).toContain("Tour da interface");
    expect(html).toContain("Primeiro treino");
  });
});
