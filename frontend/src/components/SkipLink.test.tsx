/**
 * The test setup has no DOM, so the link is rendered to markup and App.tsx is read
 * as source for the one thing a render cannot show here: the link comes before the
 * header (so it is the first focusable element) and the target exists.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { SkipLink } from "./SkipLink";

const render = (lang: "pt" | "en") =>
  renderToStaticMarkup(
    <LanguageContext.Provider
      value={{
        lang,
        t: lang === "pt" ? pt : en,
        locale: lang === "pt" ? "pt-BR" : "en-US",
        setLang: () => {},
      }}
    >
      <SkipLink />
    </LanguageContext.Provider>,
  );

const app = readFileSync(join(__dirname, "..", "App.tsx"), "utf-8");

describe("SkipLink", () => {
  it.each([
    ["pt", pt.a11y.skipToContent],
    ["en", en.a11y.skipToContent],
  ] as const)("is a link to #vf-main with the %s text", (lang, text) => {
    const html = render(lang);
    expect(html).toContain('href="#vf-main"');
    expect(html).toContain(`>${text}</a>`);
  });

  it("has a text in each language", () => {
    expect(pt.a11y.skipToContent).not.toBe(en.a11y.skipToContent);
  });

  it("comes before the header, and main#vf-main can take focus", () => {
    expect(app.indexOf("<SkipLink />")).toBeGreaterThan(0);
    expect(app.indexOf("<SkipLink />")).toBeLessThan(app.indexOf("<Header"));
    expect(app).toMatch(/<main\s+id="vf-main"\s+tabIndex=\{-1\}/);
  });
});
