import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import { taskDefinitions } from "../types/tasks";
import type { JsonSchema } from "../types/schema";
import { parseYamlToConfig, reviewImportedConfig, serializeConfigToYaml } from "../lib/yaml-config";
import { ParamPanel } from "./ParamPanel";

// Pydantic emits Optional[CrossValidationConfig] as anyOf, not a direct $ref.
const schema: JsonSchema = {
  type: "object",
  properties: {
    block: { type: "string" },
    cross_validation: { anyOf: [{ $ref: "#/$defs/CrossValidationConfig" }, { type: "null" }] },
  },
  $defs: {
    CrossValidationConfig: {
      type: "object",
      properties: {
        n_folds: { type: "integer", default: 5 },
        stratified: { type: "boolean", default: true },
        shuffle: { type: "boolean", default: true },
        fold_seed: { type: "integer", default: 42 },
        group_manifest: { anyOf: [{ type: "string" }, { type: "null" }], default: null },
      },
    },
  },
};
const formData = {
  block: "cross_validation",
  cross_validation: { n_folds: 5, stratified: true, shuffle: true, fold_seed: 42, group_manifest: "fold-groups.csv" },
};

describe("classification K-fold group configuration", () => {
  it("renders the config-bound CSV field with the real nullable Pydantic schema", () => {
    const html = renderToStaticMarkup(
      <LanguageContext.Provider value={{ lang: "pt", t: pt, locale: "pt-BR", setLang: () => {} }}>
        <ParamPanel task={taskDefinitions(pt)[0]} schema={schema} formData={formData} setFormData={() => {}} validationErrors={[]} />
      </LanguageContext.Provider>,
    );
    expect(html).toContain('value="fold-groups.csv"');
    expect(html.match(/CSV de grupos \(opcional\)/g)).toHaveLength(2);
    expect(html).toContain('data-testid="classification-cross-validation"');
  });

  it("keeps the group CSV through YAML export and reviewed import", () => {
    const yaml = serializeConfigToYaml(formData);
    expect(yaml).toContain("group_manifest: fold-groups.csv");
    const parsed = parseYamlToConfig(yaml);
    expect(parsed).toEqual(formData);
    const reviewed = reviewImportedConfig(pt, parsed, schema, schema.$defs);
    expect(reviewed.issues).toEqual([]);
    expect(reviewed.data).toEqual(formData);
  });
});

// A schema with the fields each step draws, enough to place them.
const stepSchema: JsonSchema = {
  type: "object",
  properties: {
    name: { type: "string", default: "experiment_001" },
    task: { type: "string", enum: ["binary", "multiclass"], default: "multiclass" },
    block: { type: "string", default: "classification" },
    model: { $ref: "#/$defs/ModelConfig" },
    training: { $ref: "#/$defs/TrainingConfig" },
    transfer_learning: { $ref: "#/$defs/TransferLearningConfig" },
    grid_search: { type: "object" },
  },
  $defs: {
    ModelConfig: {
      type: "object",
      properties: {
        name: { type: "string", enum: ["resnet18", "resnet50"], default: "resnet18" },
        num_classes: { type: "integer", default: 2 },
        pretrained: { type: "boolean", default: true },
      },
    },
    TrainingConfig: {
      type: "object",
      properties: {
        epochs: { type: "integer", default: 10 },
        learning_rate: { type: "number", default: 0.001 },
      },
    },
    TransferLearningConfig: {
      type: "object",
      properties: {
        mode: { type: "string", enum: ["feature_extraction", "fine_tuning"], default: "feature_extraction" },
        backbone_lr_multiplier: { type: "number", default: 0.1 },
      },
    },
  },
};

function renderSteps(data: Record<string, unknown>): string {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={{ lang: "pt", t: pt, locale: "pt-BR", setLang: () => {} }}>
      <ParamPanel task={taskDefinitions(pt)[0]} schema={stepSchema} formData={data} setFormData={() => {}} validationErrors={[]} />
    </LanguageContext.Provider>,
  );
}

/** The markup of one step's section. */
function section(html: string, step: string): string {
  const start = html.indexOf(`<section data-step="${step}"`);
  expect(start).toBeGreaterThanOrEqual(0);
  // Up to the next step, not the first </section>: a card may hold its own.
  const next = html.indexOf("<section data-step=", start + 1);
  return html.slice(start, next === -1 ? html.indexOf('<footer class="vf-step-foot"', start) : next);
}

const baseForm = {
  name: "experiment_001",
  task: "multiclass",
  block: "classification",
  model: { name: "resnet18", num_classes: 3, pretrained: true },
  training: { epochs: 10, learning_rate: 0.001 },
  data: { base_dir: "" },
};

describe("classification steps (ADR-156)", () => {
  it("splits the configuration into the six steps", () => {
    const html = renderSteps(baseForm);
    for (const step of ["data", "strategy", "model", "training", "image", "review"]) {
      expect(html).toContain(`data-step="${step}"`);
    }
  });

  it("keeps the training tour target inside the training step", () => {
    const html = renderSteps(baseForm);
    expect(section(html, "training")).toContain('data-tour="training"');
    expect(section(html, "data")).toContain('data-tour="dataset"');
  });

  it("shows the experiment name only on the review step", () => {
    const html = renderSteps(baseForm);
    const label = "Nome do experimento";
    expect(section(html, "review")).toContain(label);
    for (const step of ["data", "strategy", "model", "training", "image"]) {
      expect(section(html, step)).not.toContain(label);
    }
  });

  it("offers transfer learning in Modelo, not as a strategy", () => {
    const html = renderSteps(baseForm);
    expect(section(html, "strategy")).not.toContain("Transfer learning");
    expect(section(html, "model")).toContain("Transfer learning");
  });

  it("shows an imported transfer-learning block as on in Modelo and Treino simples as the strategy", () => {
    const html = renderSteps({
      ...baseForm,
      block: "transfer_learning",
      transfer_learning: { mode: "fine_tuning", backbone_lr_multiplier: 0.1 },
    });
    const model = section(html, "model");
    expect(model).toMatch(/role="switch" aria-checked="true"/);
    expect(model).toContain("// transfer learning");
    expect(section(html, "strategy")).toMatch(/background:var\(--accent-soft\)[^>]*>Treino simples</);
  });

  it("locks the transfer-learning switch while a multi-run strategy is chosen", () => {
    const html = renderSteps({ ...baseForm, block: "grid_search", grid_search: { hyperparameters: {} } });
    const model = section(html, "model");
    expect(model).toMatch(/<fieldset disabled=""/);
    expect(model).toContain("um bloco por vez");
  });
});
