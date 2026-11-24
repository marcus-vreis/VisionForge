import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { LanguageContext, type LanguageContextValue } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { BUILTIN_MODEL } from "../lib/model-source";
import { CustomModelFieldView, type CustomModelsLoad } from "./CustomModelField";

const noop = () => {};
const PT: LanguageContextValue = { lang: "pt", t: pt, locale: "pt-BR", setLang: noop };
const EN: LanguageContextValue = { lang: "en", t: en, locale: "en-US", setLang: noop };

function view(load: CustomModelsLoad, models: string[], value: string, dict = PT): string {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={dict}>
      <CustomModelFieldView load={load} models={models} value={value} onChange={noop} />
    </LanguageContext.Provider>,
  );
}

describe("the custom-model field of the Model step (ADR-048)", () => {
  it("draws nothing until the list is read, nor when it cannot be", () => {
    expect(view("loading", [], BUILTIN_MODEL)).toBe("");
    expect(view("failed", [], BUILTIN_MODEL)).toBe("");
  });

  it("says how to add a model instead of offering an empty list", () => {
    const markup = view("ready", [], BUILTIN_MODEL);
    expect(markup).toContain('data-custom-model="none"');
    expect(markup).toContain("<code>user_models/</code>");
    expect(markup).toContain("<code>docs/custom/MODELS.md</code>");
    expect(markup).not.toContain("aria-expanded");
  });

  it("says it in English too", () => {
    const markup = view("ready", [], BUILTIN_MODEL, EN);
    expect(markup).toContain("None yet.");
  });

  it("offers the registered models with the builtin architecture chosen", () => {
    const markup = view("ready", ["my_net"], BUILTIN_MODEL);
    expect(markup).toContain('data-custom-model="builtin"');
    expect(markup).toContain(pt.paramPanel.customModel.builtin);
    expect(markup).toContain('aria-expanded="false"');
  });

  it("shows the chosen model of the researcher's own", () => {
    const markup = view("ready", ["my_net"], "my_net");
    expect(markup).toContain('data-custom-model="custom"');
    expect(markup).toContain(">my_net<");
    expect(markup).not.toContain(pt.paramPanel.customModel.builtin);
  });

  it("keeps the selector for a model a YAML names when the folder registers none", () => {
    const markup = view("ready", [], "gone_net");
    expect(markup).toContain('data-custom-model="custom"');
    expect(markup).toContain(">gone_net<");
  });
});
