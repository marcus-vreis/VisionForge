import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { DvcRepositoryState } from "../api/client";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { canPrepare, repositoryNotes, resultLines } from "../lib/dvc-export";
import { DvcExportPanel } from "./DvcExportPanel";

const READY: DvcRepositoryState = {
  path: "C:/pesquisa/experimentos", state: "ready", remote: null, tools: { git: true, dvc: true },
};

describe("DvcExportPanel", () => {
  it.each([pt, en])("starts closed and performs no request", (t) => {
    const f = vi.fn();
    vi.stubGlobal("fetch", f);
    try {
      const html = renderToStaticMarkup(<LanguageContext.Provider value={{ lang: "pt", t, locale: "pt-BR", setLang: () => {} }}>
        <DvcExportPanel runId="exp/20261108_000000" />
      </LanguageContext.Provider>);
      expect(html).toContain(t.dvc.open);
      expect(html).not.toContain("<section");
      expect(f).not.toHaveBeenCalled();
    } finally { vi.unstubAllGlobals(); }
  });

  it("offers to prepare a folder that is not a repository yet", () => {
    const missing = { ...READY, state: "missing" as const };

    expect(canPrepare(missing)).toBe(true);
    expect(repositoryNotes(missing, pt.dvc)).toEqual([pt.dvc.state.missing, pt.dvc.prepareHint]);
  });

  it("never prepares a file, a ready repository, or without the tools", () => {
    expect(canPrepare({ ...READY, state: "not_dir" })).toBe(false);
    expect(canPrepare(READY)).toBe(false);
    expect(canPrepare({ ...READY, state: "no_dvc", tools: { git: true, dvc: false } })).toBe(false);
    expect(repositoryNotes({ ...READY, state: "no_dvc", tools: { git: true, dvc: false } }, en.dvc))
      .toEqual([en.dvc.state.no_dvc, en.dvc.noDvc]);
  });

  it("says where the weights go in a ready repository", () => {
    expect(repositoryNotes(READY, pt.dvc)).toEqual([pt.dvc.noRemote]);
    expect(repositoryNotes({ ...READY, remote: "storage" }, pt.dvc))
      .toEqual(['Os pesos vão para o remoto padrão do DVC, "storage" (dvc push).']);
  });

  it("reports the short commit, the folder and the push", () => {
    const result = { folder: "runs/a/b", path: "C:/r/runs/a/b", commit: "0123456789abcdef", remote: "s", pushed: true };

    expect(resultLines(result, pt.dvc)).toEqual(["Commit 0123456 feito.", "Arquivos em C:/r/runs/a/b", pt.dvc.pushed]);
    expect(resultLines({ ...result, commit: null, pushed: false }, en.dvc))
      .toEqual([en.dvc.unchanged, "Files in C:/r/runs/a/b", en.dvc.pushFailed]);
    expect(resultLines({ ...result, pushed: null }, en.dvc)).toHaveLength(2);
  });
});
