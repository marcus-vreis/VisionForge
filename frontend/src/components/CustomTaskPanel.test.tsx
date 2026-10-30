import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import type { CustomTaskDefinition } from "../lib/custom-tasks";
import { StepNavContext } from "../lib/steps";
import type { JsonSchema } from "../types/schema";
import { CustomTaskSteps } from "./CustomTaskPanel";

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };

const task: CustomTaskDefinition = {
  key: "tarefa_teste",
  label: "Tarefa teste",
  short: "tarefa_teste",
  description: "",
  accent: "#fff",
  custom: true,
  metrics: { acc: "Acc" },
  primaryMetric: "acc",
};

const schema: JsonSchema = {
  type: "object",
  properties: {
    name: { type: "string", default: "tarefa_teste_001" },
    alpha: { type: "number", default: 0.5 },
    data: { $ref: "#/$defs/Data" },
    training: { $ref: "#/$defs/Training" },
  },
  $defs: {
    Data: { type: "object", properties: { base_dir: { type: "string", default: "" } } },
    Training: {
      type: "object",
      properties: { epochs: { type: "integer", default: 10 } },
    },
  },
};

function render(active = "data") {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={lang}>
      <StepNavContext.Provider value={{ stepFor: () => active as never, setStep: () => {} }}>
        <CustomTaskSteps
          task={task}
          schema={schema}
          formData={{ name: "tarefa_teste_001", alpha: 0.5, data: { base_dir: "" }, training: { epochs: 10 } }}
          setFormData={() => {}}
          validationErrors={[]}
          onSweep={() => {}}
          onReplicates={() => {}}
          onRemoved={() => {}}
        />
      </StepNavContext.Provider>
    </LanguageContext.Provider>,
  );
}

function section(html: string, id: string): string {
  return html.split("<section data-step=").find((p) => p.startsWith(`"${id}"`)) ?? "";
}

describe("CustomTaskPanel steps", () => {
  const html = render();

  it("shows only the steps the schema has fields for", () => {
    for (const id of ["data", "strategy", "training", "review"]) {
      expect(html).toContain(`<section data-step="${id}"`);
    }
    expect(html).not.toContain('<section data-step="model"');
    expect(html).not.toContain('<section data-step="image"');
  });

  it("spreads the schema blocks over their steps", () => {
    expect(section(html, "data")).toContain("base dir");
    expect(section(html, "data")).not.toContain("epochs");
    // The task's own scalars land with the training block.
    expect(section(html, "training")).toContain("epochs");
    expect(section(html, "training")).toContain("alpha");
  });

  it("keeps the name and the manage card on the review step", () => {
    expect(section(html, "review")).toContain(pt.experimentHeader.nameLabel);
    expect(section(html, "review")).toContain(pt.customTaskManage.options);
    expect(section(html, "training")).not.toContain(pt.experimentHeader.nameLabel);
  });
});
