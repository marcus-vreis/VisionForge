import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { makeDefaultRegressionForm } from "../lib/regression-models";
import { StepNavContext } from "../lib/steps";
import { ComparisonCost, ReplicatedComparisonCard } from "./ReplicatedComparisonCard";
import { RegressionPanel } from "./RegressionPanel";

const withLang = (lang: "pt" | "en", node: React.ReactNode) => {
  const dict = lang === "pt" ? pt : en;
  return renderToStaticMarkup(
    <LanguageContext.Provider
      value={{ lang, t: dict, locale: lang === "pt" ? "pt-BR" : "en-US", setLang: () => {} }}
    >
      {node}
    </LanguageContext.Provider>,
  );
};

const card = (
  <ReplicatedComparisonCard
    metrics={[{ value: "r2", label: "R²" }]}
    pathHints={["training.learning_rate", "model.name"]}
    accent="#fff"
    onReplicatedComparison={() => {}}
  />
);

describe("ReplicatedComparisonCard (ADR-193)", () => {
  it.each(["pt", "en"] as const)(
    "shows the cost and the power warning for its default 5 seeds (%s)",
    (lang) => {
      const dict = lang === "pt" ? pt : en;
      const html = withLang(lang, card);

      expect(html).toContain(dict.replicatedComparisonCard.title);
      // The form alone until a value is typed: one variant, five seeds.
      expect(html).toContain(dict.replicatedComparisonCard.cost(1, 5, 5));
      expect(html).toContain(dict.replicatedComparisonCard.power(5, "0.0625", "0.05"));
      // Why it cannot run yet stands where the run summary would.
      expect(html).toContain(dict.replicatedComparisonForm.needPath);
    },
  );

  it("labels every field it renders", () => {
    const html = withLang("pt", card);
    const ids = [...html.matchAll(/<input[^>]*id="([^"]+)"/g)].map((m) => m[1]);

    expect(ids.length).toBeGreaterThanOrEqual(3);
    for (const id of ids) expect(html).toContain(`for="${id}"`);
    expect(html).toContain(pt.replicatedComparisonCard.parameter);
    expect(html).toContain(pt.replicatedComparisonCard.values);
    expect(html).toContain(pt.replicatedComparisonCard.count);
  });

  it("warns with fewer than 6 seeds under Wilcoxon", () => {
    const html = withLang("pt", <ComparisonCost variants={3} seeds={5} />);

    expect(html).toContain(pt.replicatedComparisonCard.cost(3, 5, 15));
    expect(html).toContain('role="note"');
    expect(html).toContain(pt.replicatedComparisonCard.power(5, "0.0625", "0.05"));
  });

  it("keeps quiet from 6 seeds on, where significance is reachable", () => {
    const html = withLang("en", <ComparisonCost variants={4} seeds={6} />);

    expect(html).toContain(en.replicatedComparisonCard.cost(4, 6, 24));
    expect(html).not.toContain('role="note"');
  });
});

describe("the comparison on a task panel", () => {
  it("is offered on Regressão's strategy step", () => {
    const html = withLang(
      "pt",
      <StepNavContext.Provider value={{ stepFor: () => "data", setStep: () => {} }}>
        <RegressionPanel
          formData={makeDefaultRegressionForm()}
          setFormData={() => {}}
          accent="#fff"
          validationErrors={[]}
          onReplicatedComparison={() => {}}
        />
      </StepNavContext.Provider>,
    );

    expect(html).toContain(`>${pt.taskPanel.replicatedComparison}<`);
  });
});
