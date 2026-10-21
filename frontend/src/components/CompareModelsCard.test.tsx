import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { anomalyModels, makeDefaultAnomalyForm } from "../lib/anomaly-models";
import { AnomalyPanel } from "./AnomalyPanel";
import { CompareModelsCard } from "./CompareModelsCard";

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

describe("CompareModelsCard", () => {
  it.each(["pt", "en"] as const)(
    "starts with every architecture picked and says it is a test-set report (%s)",
    (lang) => {
      const dict = lang === "pt" ? pt : en;
      const html = withLang(
        lang,
        <CompareModelsCard
          models={anomalyModels(dict)}
          metrics={[{ value: "auroc", label: "AUROC" }]}
          note={dict.anomalyPanel.compareNote}
          accent="#f5a524"
          onCompare={() => {}}
        />,
      );
      expect(html.match(/<input type="checkbox"[^>]*checked=""/g)).toHaveLength(2);
      expect(html).toContain("Autoencoder");
      expect(html).toContain("PatchCore");
      expect(html).toContain(dict.compareModelsCard.summary(2));
      expect(html).toContain(dict.compareModelsCard.title);
    },
  );

  it("words the anomaly note as a test-set report, not a selection", () => {
    expect(pt.anomalyPanel.compareNote).toContain("conjunto de teste");
    expect(pt.anomalyPanel.compareNote).toContain("não uma seleção");
    expect(en.anomalyPanel.compareNote).toContain("test set");
    expect(en.anomalyPanel.compareNote).toContain("not a selection");
  });

  it("cannot run with fewer than two architectures", () => {
    const html = withLang(
      "pt",
      <CompareModelsCard
        models={[{ value: "autoencoder", label: "Autoencoder" }]}
        metrics={[{ value: "auroc", label: "AUROC" }]}
        note="n"
        accent="#f5a524"
        onCompare={() => {}}
      />,
    );
    expect(html).toContain(pt.compareModelsCard.blocked);
  });
});

describe("AnomalyPanel strategies (ADR-131)", () => {
  it("offers the architecture comparison and no sweep", () => {
    const html = withLang(
      "pt",
      <AnomalyPanel
        formData={makeDefaultAnomalyForm()}
        setFormData={() => {}}
        accent="#f5a524"
        validationErrors={[]}
      />,
    );
    expect(html).toContain(pt.anomalyPanel.compareLabel);
    expect(html).toContain(pt.taskPanel.replicates);
    expect(html).not.toContain(`>${pt.taskPanel.sweep}<`);
  });
});
