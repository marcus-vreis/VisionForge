import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import { makeDefaultRegressionForm } from "../lib/regression-models";
import { StepNavContext } from "../lib/steps";
import { RegressionPanel } from "./RegressionPanel";

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };

function render(errors: { field: string[]; message: string }[] = []) {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={lang}>
      <StepNavContext.Provider value={{ stepFor: () => "data", setStep: () => {} }}>
        <RegressionPanel
          formData={makeDefaultRegressionForm()}
          setFormData={() => {}}
          accent="#fff"
          validationErrors={errors}
        />
      </StepNavContext.Provider>
    </LanguageContext.Provider>,
  );
}

/** The markup of one `<section data-step>`. */
function section(html: string, id: string): string {
  const parts = html.split("<section data-step=");
  return parts.find((p) => p.startsWith(`"${id}"`)) ?? "";
}

describe("RegressionPanel steps", () => {
  const html = render();

  it("renders the six steps", () => {
    for (const id of ["data", "strategy", "model", "training", "image", "review"]) {
      expect(html).toContain(`<section data-step="${id}"`);
    }
  });

  it("keeps the tour targets in their steps", () => {
    expect(section(html, "training")).toContain('data-tour="training"');
    expect(section(html, "data")).toContain('data-tour="dataset"');
    expect(section(html, "data")).toContain('data-tour="dataset-stats"');
  });

  it("puts the experiment name only on the review step", () => {
    const label = pt.experimentHeader.nameLabel;
    expect(section(html, "review")).toContain(label);
    for (const id of ["data", "strategy", "model", "training", "image"]) {
      expect(section(html, id)).not.toContain(label);
    }
  });

  it("moves image size out of the dataset step", () => {
    expect(section(html, "image")).toContain(pt.taskPanel.dataset.imageSize);
    expect(section(html, "data")).not.toContain(pt.taskPanel.dataset.imageSize);
  });

  it("lists server errors in the review instead of a top banner", () => {
    const withError = render([{ field: ["body", "data", "base_dir"], message: "obrigatório" }]);
    expect(section(withError, "review")).toContain("obrigatório");
    expect(withError).toContain(">1 erro<");
  });
});
