import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { GitHubConnect, GitHubExportPanel, GitHubOAuthReturn } from "./GitHubExportPanel";

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

  it.each([pt, en])("offers \"Conectar com GitHub\" only when the server does", (t) => {
    const render = (offered: boolean, busy = false) => renderToStaticMarkup(
      <LanguageContext.Provider value={{ lang: "pt", t, locale: "pt-BR", setLang: () => {} }}>
        <GitHubConnect offered={offered} busy={busy} onConnect={() => {}} />
      </LanguageContext.Provider>);
    expect(render(false)).toBe("");
    const html = render(true);
    expect(html).toContain(t.github.oauthConnect);
    expect(html).toContain("repo");
    expect(render(true, true)).toContain("disabled");
  });

  it.each([pt, en])("says once how the page came back from GitHub", (t) => {
    const render = (result: "connected" | "failed" | null) => renderToStaticMarkup(
      <LanguageContext.Provider value={{ lang: "pt", t, locale: "pt-BR", setLang: () => {} }}>
        <GitHubOAuthReturn result={result} />
      </LanguageContext.Provider>);
    expect(render(null)).toBe("");
    expect(render("connected")).toContain(t.github.oauthConnected);
    expect(render("connected")).toContain('role="status"');
    expect(render("failed")).toContain(t.github.oauthFailed);
    expect(render("failed")).toContain('role="alert"');
  });

  it("names the folder in the repository", () => {
    expect(pt.github.folder("runs/gatos/1")).toBe("Pasta no repositório: runs/gatos/1");
  });
});
