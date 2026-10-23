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
