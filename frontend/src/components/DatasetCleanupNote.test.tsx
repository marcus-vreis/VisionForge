import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import type { DatasetCleanup } from "../types/run";
import { DatasetCleanupNote } from "./DatasetCleanupNote";

const ptLang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };
const enLang = { lang: "en" as const, t: en, locale: "en-US", setLang: () => {} };

const DELETED: DatasetCleanup = {
  dataset: "beans",
  path: "datasets/beans",
  size_bytes: 3 * 1024 ** 2,
  at: "2026-11-25T10:00:00-03:00",
  status: "deleted",
  error: null,
};
const FAILED: DatasetCleanup = {
  ...DELETED,
  dataset: "apples",
  path: "datasets/apples",
  status: "failed",
  error: "Access is denied",
};

function render(
  entries: DatasetCleanup[] | undefined,
  lang: typeof ptLang | typeof enLang = ptLang,
): string {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={lang}>
      <DatasetCleanupNote entries={entries} />
    </LanguageContext.Provider>,
  );
}

describe("the datasets deleted after a run (ADR-249)", () => {
  it("shows nothing when the run deleted nothing", () => {
    expect(render(undefined)).toBe("");
    expect(render([])).toBe("");
  });

  it("says which dataset went, with its size", () => {
    const markup = render([DELETED]);
    expect(markup).toContain('role="status"');
    expect(markup).toContain('data-cleanup-status="deleted"');
    expect(markup).toContain(pt.datasetCleanup.deleted("beans", "3,0 MB"));
  });

  it("says a deletion failed, with the server's reason", () => {
    const markup = render([DELETED, FAILED]);
    expect(markup).toContain(pt.datasetCleanup.failed("apples", "Access is denied"));
    expect(markup.match(/data-cleanup-status=/g)).toHaveLength(2);
  });

  it("speaks English too", () => {
    expect(render([DELETED], enLang)).toContain(en.datasetCleanup.deleted("beans", "3.0 MB"));
  });
});
