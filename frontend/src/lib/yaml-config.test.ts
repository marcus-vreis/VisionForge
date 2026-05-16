import { describe, expect, it } from "vitest";
import {
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
});

describe("parseYamlToConfig", () => {
  it("throws YamlParseError on malformed YAML", () => {
    expect(() => parseYamlToConfig("{ bad: yaml: here:")).toThrow(YamlParseError);
  });

  it("throws YamlParseError with descriptive message on malformed YAML", () => {
    try {
      parseYamlToConfig("{ bad: yaml: here:");
    } catch (e) {
      expect(e).toBeInstanceOf(YamlParseError);
      expect((e as YamlParseError).message).toMatch(/YAML parse error/);
    }
  });

  it("throws YamlParseError when root is a scalar", () => {
    expect(() => parseYamlToConfig("just a string")).toThrow(YamlParseError);
  });

  it("throws YamlParseError when root is a list", () => {
    expect(() => parseYamlToConfig("- item1\n- item2")).toThrow(YamlParseError);
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
    expect(validateParsedConfig(data, SIMPLE_SCHEMA)).toEqual([]);
  });

  it("flags a missing required top-level field", () => {
    const data = { task: "binary", model: { name: "resnet50", num_classes: 1 } };
    const errors = validateParsedConfig(data, SIMPLE_SCHEMA);
    expect(errors.some((e) => e.field[0] === "name")).toBe(true);
  });

  it("flags a missing required nested field", () => {
    const data = { name: "test", task: "binary", model: { name: "resnet50" } };
    const errors = validateParsedConfig(data, SIMPLE_SCHEMA);
    expect(errors.some((e) => e.field.join(".") === "model.num_classes")).toBe(true);
  });

  it("flags an invalid enum value", () => {
    const data = { name: "test", task: "detection", model: { name: "resnet50", num_classes: 1 } };
    const errors = validateParsedConfig(data, SIMPLE_SCHEMA);
    expect(errors.some((e) => e.field[0] === "task")).toBe(true);
  });

  it("does NOT flag task/num_classes cross-field mismatch (server-side only)", () => {
    // task=binary but num_classes=5 — client validation must pass this through
    const data = { name: "test", task: "binary", model: { name: "resnet50", num_classes: 5 } };
    const errors = validateParsedConfig(data, SIMPLE_SCHEMA);
    expect(errors).toEqual([]);
  });
});
