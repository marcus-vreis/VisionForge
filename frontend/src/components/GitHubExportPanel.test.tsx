import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { GitHubExportPanel } from "./GitHubExportPanel";

describe("GitHubExportPanel", () => {
  it.each([pt, en])("starts closed and performs no request", (t) => {
    const f = vi.fn();
    vi.stubGlobal("fetch", f);
    try {
      const html = renderToStaticMarkup(<LanguageContext.Provider value={{ lang: "pt", t, locale: "pt-BR", setLang: () => {} }}>
        <GitHubExportPanel runId="exp/20261108_000000" />
      </LanguageContext.Provider>);
      expect(html).toContain(t.github.open);
      expect(html).not.toContain("<section");
      expect(html).not.toContain('type="password"');
      expect(f).not.toHaveBeenCalled();
    } finally { vi.unstubAllGlobals(); }
  });

  it("asks for a token limited to the experiments repository", () => {
    expect(pt.github.tokenHint).toContain("Contents");
    expect(en.github.tokenHint).toContain("Fine-grained");
  });

  it("names the folder in the repository", () => {
    expect(pt.github.folder("runs/gatos/1")).toBe("Pasta no repositório: runs/gatos/1");
  });
});
