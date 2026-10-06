import { describe, expect, it } from "vitest";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import {
  importConfigFromYaml,
  omitInvalidLeaves,
  YamlParseError,
  parseYamlToConfig,
  sanitizeForExport,
  serializeConfigToYaml,
  validateParsedConfig,
} from "./yaml-config";
import type { JsonSchema } from "../types/schema";

const BASELINE_CONFIG: Record<string, unknown> = {
  name: "resnet50_baseline",
  task: "binary",
  model: {
    name: "resnet50",
    num_classes: 1,
    pretrained: true,
    weights_path: null,
  },
  training: {
    learning_rate: 0.0001,
    epochs: 100,
    batch_size: 16,
    early_stopping_patience: 10,
    optimizer: "adam",
    weight_decay: 0.0,
    seed: 42,
  },
  data: {
    base_dir: "datasets/USK-Coffee_Binary",
    train_dir: "train",
    val_dir: "val",
    test_dir: "test",
    num_workers: 4,
    pin_memory: true,
  },
  output: {
    models_dir: "outputs/models",
    graphics_dir: "outputs/graphics",
    logs_dir: "outputs/logs",
    reports_dir: "outputs/reports",
  },
};

/** The YamlParseError a text raises, or null when it parses. */
const parseError = (text: string) => {
  try {
    parseYamlToConfig(text);
  } catch (e) {
    return e instanceof YamlParseError ? e : null;
  }
  return null;
};

describe("sanitizeForExport", () => {
  it("drops undefined keys, keeps null keys", () => {
    const input = { a: 1, b: undefined, c: null, d: "x" };
    const result = sanitizeForExport(input) as Record<string, unknown>;
    expect("b" in result).toBe(false);
    expect(result.c).toBeNull();
    expect(result.a).toBe(1);
  });

  it("recursively drops undefined in nested objects", () => {
    const input = { model: { name: "resnet50", weights_path: undefined } };
    const result = sanitizeForExport(input) as { model: Record<string, unknown> };
    expect("weights_path" in result.model).toBe(false);
    expect(result.model.name).toBe("resnet50");
  });

  it("preserves null in nested objects", () => {
    const input = { model: { weights_path: null } };
    const result = sanitizeForExport(input) as { model: Record<string, unknown> };
    expect(result.model.weights_path).toBeNull();
  });
});

describe("serializeConfigToYaml + parseYamlToConfig round-trip", () => {
  it("round-trips the baseline config exactly", () => {
    const yaml = serializeConfigToYaml(BASELINE_CONFIG);
    const parsed = parseYamlToConfig(yaml);
    expect(parsed).toEqual(BASELINE_CONFIG);
  });

  it("preserves null fields through the round-trip (weights_path: null)", () => {
    const yaml = serializeConfigToYaml(BASELINE_CONFIG);
    const parsed = parseYamlToConfig(yaml) as { model: Record<string, unknown> };
    expect(parsed.model.weights_path).toBeNull();
    expect(parsed.model.weights_path).not.toBe(undefined);
    expect(parsed.model.weights_path).not.toBe("None");
  });

  it("strips undefined fields on export and round-trips cleanly", () => {
    const withUndefined = {
      ...BASELINE_CONFIG,
      training: { ...(BASELINE_CONFIG.training as object), epochs: undefined },
    };
    const yaml = serializeConfigToYaml(withUndefined);
    const parsed = parseYamlToConfig(yaml) as { training: Record<string, unknown> };
    expect("epochs" in parsed.training).toBe(false);
  });

  it("preserves data.preprocessing.steps with custom params", () => {
    // Regression guard: until the PreprocessingPanel became controlled, this
    // round-trip wasn't even possible — the pipeline lived in panel state and
    // never reached formData. Confirms YAML export reproduces the pipeline.
    const cfg = {
      ...BASELINE_CONFIG,
      data: {
        ...(BASELINE_CONFIG.data as object),
        preprocessing: {
          steps: [
            { kind: "gaussian_blur", radius: 1.5 },
            { kind: "grayscale" },
            { kind: "wavelet", band: "LL" },
          ],
        },
      },
    };
    const yaml = serializeConfigToYaml(cfg);
    const parsed = parseYamlToConfig(yaml) as {
      data: {
        preprocessing: { steps: Array<Record<string, unknown>> };
      };
    };
    expect(parsed.data.preprocessing.steps).toEqual([
      { kind: "gaussian_blur", radius: 1.5 },
      { kind: "grayscale" },
      { kind: "wavelet", band: "LL" },
    ]);
  });
});

describe("parseYamlToConfig", () => {
  it("throws YamlParseError on malformed YAML", () => {
    expect(() => parseYamlToConfig("{ bad: yaml: here:")).toThrow(YamlParseError);
  });

  it("tells a syntax error apart, carrying js-yaml's own message", () => {
    const err = parseError("{ bad: yaml: here:");
    expect(err?.kind).toBe("syntax");
    // No wording of ours in front of the parser's message.
    expect(err?.message).not.toMatch(/YAML parse error/);
    expect(err?.message.length).toBeGreaterThan(0);
  });

  it("throws YamlParseError when root is a scalar", () => {
    expect(() => parseYamlToConfig("just a string")).toThrow(YamlParseError);
    expect(parseError("just a string")?.kind).toBe("notMapping");
  });

  it("throws YamlParseError when root is a list", () => {
    expect(() => parseYamlToConfig("- item1\n- item2")).toThrow(YamlParseError);
    expect(parseError("- item1\n- item2")?.kind).toBe("notMapping");
  });
});

describe("validateParsedConfig", () => {
  const SIMPLE_SCHEMA: JsonSchema = {
    type: "object",
    properties: {
      name: { type: "string" },
      task: { type: "string", enum: ["binary", "multiclass"] },
      model: {
        type: "object",
        properties: {
          num_classes: { type: "integer" },
          name: { type: "string" },
        },
        required: ["name", "num_classes"],
      },
    },
    required: ["name", "task", "model"],
  };

  it("returns no errors for a valid config", () => {
    const data = { name: "test", task: "binary", model: { name: "resnet50", num_classes: 1 } };
    expect(validateParsedConfig(pt, data, SIMPLE_SCHEMA)).toEqual([]);
  });

  it("flags a missing required top-level field", () => {
    const data = { task: "binary", model: { name: "resnet50", num_classes: 1 } };
    const errors = validateParsedConfig(pt, data, SIMPLE_SCHEMA);
    expect(errors.some((e) => e.field[0] === "name")).toBe(true);
  });

  it("flags a missing required nested field", () => {
    const data = { name: "test", task: "binary", model: { name: "resnet50" } };
    const errors = validateParsedConfig(pt, data, SIMPLE_SCHEMA);
    expect(errors.some((e) => e.field.join(".") === "model.num_classes")).toBe(true);
  });

  it("flags an invalid enum value", () => {
    const data = { name: "test", task: "detection", model: { name: "resnet50", num_classes: 1 } };
    const errors = validateParsedConfig(pt, data, SIMPLE_SCHEMA);
    expect(errors.some((e) => e.field[0] === "task")).toBe(true);
  });

  it("words each problem in the language it is given", () => {
    const data = { task: "detection", model: { name: "resnet50", num_classes: "many" } };
    const say = (dict: typeof pt) =>
      Object.fromEntries(
        validateParsedConfig(dict, data, {
          ...SIMPLE_SCHEMA,
          properties: {
            ...SIMPLE_SCHEMA.properties,
            model: {
              type: "object",
              properties: { num_classes: { type: "integer" }, name: { type: "string" } },
            },
          },
        }).map((e) => [e.field.join("."), e.message]),
      );
    expect(say(pt)).toEqual({
      name: "Campo obrigatório ausente.",
      task: "Deve ser um destes: binary, multiclass.",
      "model.num_classes": "Esperado um inteiro.",
    });
    expect(say(en)).toEqual({
      name: "Required field is missing.",
      task: "Must be one of: binary, multiclass.",
      "model.num_classes": "Expected an integer.",
    });
  });

  it("does NOT flag task/num_classes cross-field mismatch (server-side only)", () => {
    // task=binary but num_classes=5 — client validation must pass this through
    const data = { name: "test", task: "binary", model: { name: "resnet50", num_classes: 5 } };
    const errors = validateParsedConfig(pt, data, SIMPLE_SCHEMA);
    expect(errors).toEqual([]);
  });

  describe("nullable fields (anyOf with a null branch)", () => {
    // Pydantic's `Optional[X]`: `weights_path: str | None = None`. The export
    // writes the null out, so importing our own file has to accept it.
    const NULLABLE: JsonSchema = {
      type: "object",
      properties: {
        weights_path: { anyOf: [{ type: "string" }, { type: "null" }] },
        epochs: { anyOf: [{ type: "integer" }, { type: "null" }] },
        grid: { anyOf: [{ $ref: "#/$defs/Grid" }, { type: "null" }] },
        mode: { anyOf: [{ type: "string", enum: ["a", "b"] }, { type: "null" }] },
      },
    };
    const defs: Record<string, JsonSchema> = {
      Grid: { type: "object", properties: { n: { type: "integer" } } },
    };
    const issues = (data: Record<string, unknown>) =>
      validateParsedConfig(pt, data, NULLABLE, defs).map((e) => e.field.join("."));

    it("accepts an explicit null", () => {
      expect(issues({ weights_path: null, epochs: null, grid: null, mode: null })).toEqual([]);
    });

    it("still checks the type of a value that is not null", () => {
      expect(issues({ weights_path: 5, epochs: "ten", grid: 3, mode: "c" })).toEqual([
        "weights_path",
        "epochs",
        "grid",
        "mode",
      ]);
    });

    it("does not let null through a field that is not nullable", () => {
      const strict: JsonSchema = { type: "object", properties: { name: { type: "string" } } };
      expect(validateParsedConfig(pt, { name: null }, strict).map((e) => e.field)).toEqual([
        ["name"],
      ]);
    });
  });
});

describe("omitInvalidLeaves", () => {
  const SCHEMA: JsonSchema = {
    type: "object",
    properties: {
      name: { type: "string" },
      task: { type: "string", enum: ["binary", "multiclass"] },
      model: {
        type: "object",
        properties: {
          name: { type: "string" },
          num_classes: { type: "integer" },
          pretrained: { type: "boolean" },
          weights_path: { anyOf: [{ type: "string" }, { type: "null" }] },
        },
        required: ["name", "num_classes"],
      },
      training: { $ref: "#/$defs/Training" },
      data: { $ref: "#/$defs/Data" },
    },
    required: ["name", "task", "model", "data"],
    $defs: {
      Training: {
        type: "object",
        properties: { epochs: { type: "integer" }, learning_rate: { type: "number" } },
      },
      Data: {
        type: "object",
        properties: {
          base_dir: { type: "string" },
          train_dir: { type: "string" },
          class_names: { type: "array", items: { type: "string" } },
          sub: {
            type: "object",
            properties: { deep: { type: "string" }, keep: { type: "string" } },
          },
        },
        required: ["base_dir"],
      },
    },
  };

  /** parse → validate → omit, the way the import runs it. */
  const run = (yaml: string) => {
    const parsed = parseYamlToConfig(yaml);
    const issues = validateParsedConfig(pt, parsed, SCHEMA, SCHEMA.$defs);
    const { data: kept, omitted } = omitInvalidLeaves(parsed, issues);
    return { parsed, issues, kept, omitted };
  };

  it("drops a numeric experiment name and keeps its siblings", () => {
    const { kept } = run("name: 5\ntask: binary\n");
    expect(kept).toEqual({ task: "binary" });
    expect("name" in kept).toBe(false);
  });

  it("drops a numeric data.base_dir and keeps the valid sibling leaves", () => {
    const { kept } = run("name: ok\ndata:\n  base_dir: 7\n  train_dir: train\n");
    expect(kept).toEqual({ name: "ok", data: { train_dir: "train" } });
  });

  it("follows a path through $ref'd sections down to the leaf", () => {
    const { kept } = run(
      "data:\n  base_dir: x\n  sub:\n    deep: 1\n    keep: fine\ntraining:\n  epochs: ten\n  learning_rate: 0.1\n",
    );
    expect(kept).toEqual({
      data: { base_dir: "x", sub: { keep: "fine" } },
      training: { learning_rate: 0.1 },
    });
  });

  it("drops a wrong-typed leaf of every kind the validator checks", () => {
    const { kept, omitted } = run(
      [
        "name: [1, 2]",
        "task: detection",
        "model:",
        "  name: 9",
        "  num_classes: many",
        "  pretrained: 'yes'",
        "training:",
        "  epochs: 1.5e3x",
        "  learning_rate: fast",
        "data: 12",
      ].join("\n"),
    );
    expect(kept).toEqual({ model: {}, training: {} });
    expect(omitted.map((p) => p.join(".")).sort()).toEqual(
      [
        "name",
        "task",
        "model.name",
        "model.num_classes",
        "model.pretrained",
        "training.epochs",
        "training.learning_rate",
        "data",
      ].sort(),
    );
  });

  it("drops an enum value that is not one of the options", () => {
    const { kept } = run("name: ok\ntask: regression\n");
    expect(kept).toEqual({ name: "ok" });
  });

  it("keeps an explicit null on a nullable field", () => {
    const { kept } = run("name: ok\nmodel:\n  name: m\n  num_classes: 1\n  weights_path: null\n");
    expect(kept.model).toEqual({ name: "m", num_classes: 1, weights_path: null });
  });

  it("leaves arrays alone unless they are what was flagged", () => {
    const parsed = { data: { class_names: ["a", "b"], base_dir: 5 }, name: ["x"] };
    const { data: kept } = omitInvalidLeaves(parsed, [
      { field: ["data", "base_dir"] },
      { field: ["name"] },
    ]);
    expect(kept).toEqual({ data: { class_names: ["a", "b"] } });
  });

  it("drops a flagged array element without leaving a hole", () => {
    const parsed = { list: ["a", 2, "c", 4] };
    const { data: kept, omitted } = omitInvalidLeaves(parsed, [
      { field: ["list", "1"] },
      { field: ["list", "3"] },
    ]);
    expect(kept).toEqual({ list: ["a", "c"] });
    expect(omitted).toEqual([
      ["list", "1"],
      ["list", "3"],
    ]);
  });

  it("does nothing for a required field that is simply absent", () => {
    const { parsed, issues, kept, omitted } = run("name: ok\ntask: binary\n");
    expect(issues.some((i) => i.field.join(".") === "data")).toBe(true);
    expect(kept).toEqual(parsed);
    expect(omitted).toEqual([]);
    // No empty `data: {}` conjured up to hold the missing child.
    expect("data" in kept).toBe(false);
  });

  it("does not touch the object it was given", () => {
    const parsed = { name: 5, data: { base_dir: 7, train_dir: "t" } };
    const snapshot = structuredClone(parsed);
    const { data: kept } = omitInvalidLeaves(parsed, [
      { field: ["name"] },
      { field: ["data", "base_dir"] },
    ]);
    expect(parsed).toEqual(snapshot);
    expect(kept).not.toBe(parsed);
    expect(kept.data).not.toBe(parsed.data);
  });

  it("returns the same object when there is nothing to drop", () => {
    const parsed = { name: "ok" };
    expect(omitInvalidLeaves(parsed, []).data).toBe(parsed);
  });

  it("ignores a path that leads through a scalar or nowhere", () => {
    const parsed = { name: "ok", data: "text" };
    const { data: kept, omitted } = omitInvalidLeaves(parsed, [
      { field: ["name", "deeper"] },
      { field: ["nope", "x"] },
      { field: [] },
    ]);
    expect(kept).toEqual(parsed);
    expect(omitted).toEqual([]);
  });

  it("round-trip: what survives parse → validate → omit has only correctly typed leaves", () => {
    const { kept } = run(
      [
        "name: 5",
        "task: binary",
        "model:",
        "  name: resnet50",
        "  num_classes: one",
        "  pretrained: true",
        "  weights_path: null",
        "training:",
        "  epochs: 10",
        "  learning_rate: [0.1]",
        "data:",
        "  base_dir: 7",
        "  train_dir: train",
        "  class_names: [x, y]",
      ].join("\n"),
    );
    const left = validateParsedConfig(pt, kept, SCHEMA, SCHEMA.$defs);
    // Only absences are left to report; no wrong type survived.
    expect(left.map((i) => i.message)).toEqual([
      pt.yamlConfig.requiredMissing, // name
      pt.yamlConfig.requiredMissing, // model.num_classes
      pt.yamlConfig.requiredMissing, // data.base_dir
    ]);
    expect(kept).toEqual({
      task: "binary",
      model: { name: "resnet50", pretrained: true, weights_path: null },
      training: { epochs: 10 },
      data: { train_dir: "train", class_names: ["x", "y"] },
    });
  });
});

describe("importConfigFromYaml", () => {
  const file = (text: string) => new File([text], "config.yaml");

  it("hands back the parsed config", async () => {
    expect(await importConfigFromYaml(pt, file("name: run_1\n"))).toEqual({
      data: { name: "run_1" },
    });
  });

  it("reports a file that is not a YAML mapping in the language it is given", async () => {
    const asPt = await importConfigFromYaml(pt, file("- a\n- b\n"));
    const asEn = await importConfigFromYaml(en, file("- a\n- b\n"));
    expect(asPt).toEqual({ error: `Arquivo YAML inválido: ${pt.yamlConfig.notMapping}` });
    expect(asEn).toEqual({ error: `Invalid YAML file: ${en.yamlConfig.notMapping}` });
    // None of VisionForge's own English leaks into the Portuguese message.
    expect("error" in asPt && asPt.error).not.toMatch(/must contain/);
  });

  it("prefixes a syntax error once, followed by the parser's raw message", async () => {
    const bad = "{ bad: yaml: here:";
    const raw = parseError(bad)?.message ?? "";
    const asPt = await importConfigFromYaml(pt, file(bad));
    const asEn = await importConfigFromYaml(en, file(bad));
    expect(asPt).toEqual({ error: `Arquivo YAML inválido: ${raw}` });
    expect(asEn).toEqual({ error: `Invalid YAML file: ${raw}` });
    expect("error" in asEn && asEn.error).not.toMatch(/parse error/i);
  });

  it("reports a file it cannot read in the language it is given", async () => {
    const unreadable = { text: () => Promise.reject(new Error("boom")) } as unknown as File;
    expect(await importConfigFromYaml(pt, unreadable)).toEqual({
      error: "Não foi possível ler o arquivo YAML: boom",
    });
    expect(await importConfigFromYaml(en, unreadable)).toEqual({
      error: "Failed to read the YAML file: boom",
    });
  });
});
