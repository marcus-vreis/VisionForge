import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { DriveConsentPrompt } from "./DriveConsentPrompt";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { LanguageContext } from "../i18n/context";

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };

const prompt = { url: "https://accounts.google.com/x", attempt: 2, attempts: 3 };

describe("DriveConsentPrompt", () => {
  it("offers the consent page and the manual fallback, with the attempt", () => {
    const html = renderToStaticMarkup(
      <LanguageContext.Provider value={lang}>
        <DriveConsentPrompt prompt={prompt} runId="r1" />
      </LanguageContext.Provider>,
    );
    expect(html).toContain("data-drive-consent");
    expect(html).toContain(pt.trainingOverlay.driveConsentOpen);
    expect(html).toContain(pt.trainingOverlay.driveConsentDone);
    expect(html).toContain("Tentativa 2 de 3");
    // The URL carries a token in its state: it is opened, never printed.
    expect(html).not.toContain(prompt.url);
  });

  it("words the attempt in both languages", () => {
    expect(en.trainingOverlay.driveConsentBody(1, 3)).toContain("Attempt 1 of 3");
    expect(pt.trainingOverlay.driveConsentBody(1, 3)).toContain("Tentativa 1 de 3");
  });
});
