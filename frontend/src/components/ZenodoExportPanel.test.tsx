import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { ZenodoExportPanel } from "./ZenodoExportPanel";

describe("ZenodoExportPanel", () => {
  it.each([pt, en])("starts closed and performs no request", (t) => {
    const f = vi.fn();
    vi.stubGlobal("fetch", f);
    try {
      const html = renderToStaticMarkup(<LanguageContext.Provider value={{ lang: "pt", t, locale: "pt-BR", setLang: () => {} }}>
        <ZenodoExportPanel runId="exp/20261108_000000" />
      </LanguageContext.Provider>);
      expect(html).toContain(t.zenodo.open);
      expect(html).not.toContain("<section");
      expect(html).not.toContain(t.zenodo.intro);
      expect(html).not.toContain('type="password"');
      expect(f).not.toHaveBeenCalled();
    } finally { vi.unstubAllGlobals(); }
  });

  it("tells the token scope that cannot publish", () => {
    expect(pt.zenodo.tokenHint).toContain("deposit:write");
    expect(en.zenodo.tokenHint).toContain("deposit:actions");
  });

  it("words what was left out", () => {
    expect(pt.zenodo.leftOut(1)).toBe("1 item ficou de fora");
    expect(en.zenodo.leftOut(3)).toBe("3 items were left out");
  });
});
