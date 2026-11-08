import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import { en } from "../i18n/en";
import { HubExportPanel } from "./HubExportPanel";

describe("HubExportPanel", () => {
  it.each([pt, en])("starts with explicit review and performs no request", (t) => {
    const f = vi.fn();
    vi.stubGlobal("fetch", f);
    try {
      const html = renderToStaticMarkup(<LanguageContext.Provider value={{ lang: "pt", t, locale: "pt-BR", setLang: () => {} }}>
        <HubExportPanel runId="exp/20261108_000000" />
      </LanguageContext.Provider>);
      expect(html).toContain(t.hub.open);
      expect(html).not.toContain(t.hub.send);
      expect(html).not.toContain('type="password"');
      expect(f).not.toHaveBeenCalled();
    } finally { vi.unstubAllGlobals(); }
  });
});
