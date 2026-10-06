import { describe, expect, it } from "vitest";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import {
  importConfigFromYaml,
  checkImportedConfig,
  coerceNumericStrings,
  omitInvalidLeaves,
  reviewImportedConfig,
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

  describe("lists and dicts", () => {
    // These are the schema's own fragments for the two sweep blocks: the grid
    // maps a dot-path to a list of values, the random search keeps its space as
    // a free-form `dict[str, Any]`.
    const SWEEPS: JsonSchema = {
      type: "object",
      properties: {
        grid_search: {
          anyOf: [{ $ref: "#/$defs/GridSearchConfig" }, { type: "null" }],
        },
        random_search: {
          anyOf: [{ $ref: "#/$defs/RandomSearchConfig" }, { type: "null" }],
        },
        device: { $ref: "#/$defs/Device" },
        extras: { type: "object", additionalProperties: true },
        extensions: { type: "array", items: { type: "string" } },
        models: { type: "array", items: { type: "string", enum: ["a", "b"] } },
      },
    };
    const defs: Record<string, JsonSchema> = {
      GridSearchConfig: {
        type: "object",
        properties: {
          hyperparameters: {
            type: "object",
            additionalProperties: { type: "array", items: {} },
          },
        },
      },
      RandomSearchConfig: {
        type: "object",
        properties: {
          n_trials: { type: "integer" },
          search_space: { type: "object", additionalProperties: true },
        },
        required: ["n_trials"],
      },
      Device: {
        type: "object",
        properties: {
          gpu_ids: {
            anyOf: [{ type: "array", items: { type: "integer" } }, { type: "null" }],
          },
        },
      },
    };
    const issues = (data: Record<string, unknown>) =>
      validateParsedConfig(pt, data, SWEEPS, defs).map((e) => [e.field.join("."), e.message]);

    it("flags a scalar where a list is expected, in both languages", () => {
      expect(issues({ extensions: ".png" })).toEqual([["extensions", pt.yamlConfig.expectedArray]]);
      const asEn = validateParsedConfig(en, { extensions: ".png" }, SWEEPS, defs);
      expect(asEn.map((e) => e.message)).toEqual([en.yamlConfig.expectedArray]);
      expect(pt.yamlConfig.expectedArray).toBe("Esperada uma lista.");
      expect(en.yamlConfig.expectedArray).toBe("Expected a list.");
    });

    it("checks each list item against the items schema and names the index", () => {
      expect(issues({ extensions: [".png", 7, ".jpg", null] })).toEqual([
        ["extensions.1", pt.yamlConfig.expectedString],
        ["extensions.3", pt.yamlConfig.expectedString],
      ]);
      expect(issues({ models: ["a", "zzz"] })).toEqual([
        ["models.1", pt.yamlConfig.mustBeOneOf("a, b")],
      ]);
    });

    it("accepts a good list, an empty one, and a nullable list left null", () => {
      expect(issues({ extensions: [], models: ["a", "b"] })).toEqual([]);
      expect(issues({ device: { gpu_ids: [0, 1] } })).toEqual([]);
      expect(issues({ device: { gpu_ids: null } })).toEqual([]);
    });

    it("flags a nullable list that is neither a list nor null", () => {
      expect(issues({ device: { gpu_ids: "0,1" } })).toEqual([
        ["device.gpu_ids", pt.yamlConfig.expectedArray],
      ]);
      expect(issues({ device: { gpu_ids: [0, "x"] } })).toEqual([
        ["device.gpu_ids.1", pt.yamlConfig.expectedInteger],
      ]);
    });

    it("checks the values of a dict that gives an additionalProperties schema", () => {
      expect(
        issues({
          grid_search: {
            hyperparameters: {
              "training.learning_rate": [0.1, 0.01],
              "training.scheduler.kind": "cosine",
            },
          },
        }),
      ).toEqual([
        ["grid_search.hyperparameters.training.scheduler.kind", pt.yamlConfig.expectedArray],
      ]);
    });

    it("flags a dict field that is not a mapping, even when it declares no properties", () => {
      expect(issues({ grid_search: { hyperparameters: [1, 2] } })).toEqual([
        ["grid_search.hyperparameters", pt.yamlConfig.expectedObject],
      ]);
      expect(issues({ random_search: { n_trials: 3, search_space: "wide" } })).toEqual([
        ["random_search.search_space", pt.yamlConfig.expectedObject],
      ]);
    });

    it("leaves a free-form dict alone (the schema says nothing about its values)", () => {
      expect(issues({ extras: { x: 5, y: ["a"], z: "s", w: null } })).toEqual([]);
      expect(issues({ extras: [1] })).toEqual([["extras", pt.yamlConfig.expectedObject]]);
    });

    it("requires every entry of a random search space to be a mapping", () => {
      // The form reads each entry as `{type, low, high}` / `{type, options}`.
      expect(
        issues({
          random_search: {
            n_trials: 3,
            search_space: {
              ok: { type: "uniform", low: 0, high: 1 },
              nothing: null,
              scalar: 5,
              list: [1, 2],
            },
          },
        }),
      ).toEqual([
        ["random_search.search_space.nothing", pt.yamlConfig.expectedObject],
        ["random_search.search_space.scalar", pt.yamlConfig.expectedObject],
        ["random_search.search_space.list", pt.yamlConfig.expectedObject],
      ]);
    });

    describe("what the import does with them", () => {
      const load = (yaml: string) => {
        const parsed = parseYamlToConfig(yaml);
        const found = validateParsedConfig(pt, parsed, SWEEPS, defs);
        return omitInvalidLeaves(parsed, found).data;
      };

      it("keeps a scalar out of the grid axes the scheduler fields read as lists", () => {
        // `block: grid_search` + a scalar axis used to reach `.map(String)` on a string.
        const kept = load(
          [
            "grid_search:",
            "  hyperparameters:",
            "    training.scheduler.kind: cosine",
            "    training.learning_rate: [0.1, 0.01]",
          ].join("\n"),
        ) as { grid_search: { hyperparameters: Record<string, unknown> } };
        expect(kept.grid_search.hyperparameters).toEqual({
          "training.learning_rate": [0.1, 0.01],
        });
        for (const axis of Object.values(kept.grid_search.hyperparameters)) {
          expect(Array.isArray(axis)).toBe(true);
        }
      });

      it("keeps a null entry out of the random search space the rows read `.type` from", () => {
        // `random_search.search_space: {x: null}` used to reach `def.type` on null.
        const kept = load(
          [
            "random_search:",
            "  n_trials: 4",
            "  search_space:",
            "    x: null",
            "    lr: {type: log_uniform, low: 0.0001, high: 0.1}",
          ].join("\n"),
        ) as { random_search: { search_space: Record<string, unknown> } };
        expect(kept.random_search.search_space).toEqual({
          lr: { type: "log_uniform", low: 0.0001, high: 0.1 },
        });
        for (const def of Object.values(kept.random_search.search_space)) {
          expect(def).not.toBeNull();
          expect(typeof def).toBe("object");
        }
      });
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

  describe("a list is all or nothing", () => {
    // A list with an item cut out is still a list the backend accepts, and
    // training then runs on values nobody wrote: a mean with two of three
    // channels, a GPU quietly lost. So one bad item takes the whole list out,
    // and the field falls back to its default.
    it("drops the whole list when any item is flagged, reporting the list once", () => {
      const parsed = { list: ["a", 2, "c", 4], other: ["x", "y"] };
      const { data: kept, omitted } = omitInvalidLeaves(parsed, [
        { field: ["list", "1"] },
        { field: ["list", "3"] },
      ]);
      expect(kept).toEqual({ other: ["x", "y"] });
      expect(omitted).toEqual([["list"]]);
    });

    it("drops the list a flagged leaf of an item belongs to", () => {
      const parsed = { steps: [{ kind: "grayscale" }, { kind: 5 }], keep: [1] };
      const { data: kept, omitted } = omitInvalidLeaves(parsed, [
        { field: ["steps", "1", "kind"] },
      ]);
      expect(kept).toEqual({ keep: [1] });
      expect(omitted).toEqual([["steps"]]);
    });

    it("stops at the nearest list: an inner list goes, the outer one stays", () => {
      const parsed = { rows: [{ id: "a", cells: [1, "x"] }, { id: "b", cells: [3] }] };
      const { data: kept, omitted } = omitInvalidLeaves(parsed, [
        { field: ["rows", "0", "cells", "1"] },
      ]);
      expect(kept).toEqual({ rows: [{ id: "a" }, { id: "b", cells: [3] }] });
      expect(omitted).toEqual([["rows", "0", "cells"]]);
    });

    it("still drops a leaf that is not inside a list on its own", () => {
      const parsed = { data: { base_dir: 7, class_names: ["a"] } };
      const { data: kept, omitted } = omitInvalidLeaves(parsed, [
        { field: ["data", "base_dir"] },
      ]);
      expect(kept).toEqual({ data: { class_names: ["a"] } });
      expect(omitted).toEqual([["data", "base_dir"]]);
    });

    it("ignores an index that points past the end of the list", () => {
      const parsed = { list: ["a"] };
      const { data: kept, omitted } = omitInvalidLeaves(parsed, [{ field: ["list", "5"] }]);
      expect(kept).toEqual(parsed);
      expect(omitted).toEqual([]);
    });
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

  describe("a `__proto__` key in the file", () => {
    const own = (o: object, key: string) => Object.hasOwn(o, key);
    const polluted = (key: string) =>
      key in Object.prototype || key in ({} as Record<string, unknown>);

    it("stays an ordinary own key and pollutes nothing", () => {
      const parsed = parseYamlToConfig(
        [
          "name: 5",
          "task: binary",
          "__proto__:",
          "  pollutedTop: true",
          "data:",
          "  __proto__:",
          "    pollutedNested: true",
          "  base_dir: 7",
          "  train_dir: train",
        ].join("\n"),
      );
      // The parser's own guarantee, which the copy below must not undo.
      expect(own(parsed, "__proto__")).toBe(true);

      const { data: kept } = omitInvalidLeaves(parsed, [
        { field: ["name"] },
        { field: ["data", "base_dir"] },
      ]);
      const data = kept.data as Record<string, unknown>;

      expect(own(kept, "__proto__")).toBe(true);
      expect(Object.getOwnPropertyDescriptor(kept, "__proto__")?.value).toEqual({
        pollutedTop: true,
      });
      expect(own(data, "__proto__")).toBe(true);
      expect(Object.getOwnPropertyDescriptor(data, "__proto__")?.value).toEqual({
        pollutedNested: true,
      });
      // A plain `out[key] = v` would have swapped these prototypes instead.
      expect(Object.getPrototypeOf(kept)).toBe(Object.prototype);
      expect(Object.getPrototypeOf(data)).toBe(Object.prototype);
      expect(polluted("pollutedTop")).toBe(false);
      expect(polluted("pollutedNested")).toBe(false);
      // And the rest of the pruning still happened around it.
      expect("name" in kept).toBe(false);
      expect(data.train_dir).toBe("train");
      expect("base_dir" in data).toBe(false);
    });

    it("can itself be the flagged key, and is dropped like any other", () => {
      const parsed = parseYamlToConfig("task: binary\n__proto__:\n  pollutedTop: true\n");
      const { data: kept, omitted } = omitInvalidLeaves(parsed, [{ field: ["__proto__"] }]);
      expect(own(kept, "__proto__")).toBe(false);
      expect(kept).toEqual({ task: "binary" });
      expect(omitted).toEqual([["__proto__"]]);
      expect(Object.getPrototypeOf(kept)).toBe(Object.prototype);
      expect(polluted("pollutedTop")).toBe(false);
    });
  });
});

describe("reviewImportedConfig", () => {
  const SCHEMA: JsonSchema = {
    type: "object",
    properties: {
      name: { type: "string" },
      note: { type: "string" },
      epochs: { type: "integer" },
      data: {
        type: "object",
        properties: { base_dir: { type: "string" }, val_dir: { type: "string" } },
        required: ["base_dir"],
      },
    },
    required: ["name", "data"],
  };
  const review = (yaml: string) =>
    reviewImportedConfig(pt, parseYamlToConfig(yaml), SCHEMA, {});

  it("has nothing to say about a clean file", () => {
    const r = review("name: a\nepochs: 3\ndata:\n  base_dir: d\n");
    expect(r.issues).toEqual([]);
    expect(r.ignored).toBe(0);
    expect(r.missing).toBe(0);
    expect(r.data).toEqual({ name: "a", epochs: 3, data: { base_dir: "d" } });
  });

  it("an optional value of the wrong type is ignored and its field falls back to the default", () => {
    const r = review("name: a\nnote: 5\nepochs: many\ndata:\n  base_dir: d\n  val_dir: 9\n");
    expect(r.ignored).toBe(3);
    expect(r.missing).toBe(0);
    expect(r.data).toEqual({ name: "a", data: { base_dir: "d" } });
  });

  it("a required value of the wrong type has no default to fall back to: it is missing, not ignored", () => {
    const r = review("name: 5\ndata:\n  base_dir: 7\n");
    expect(r.ignored).toBe(0);
    expect(r.missing).toBe(2);
    expect(r.data).toEqual({ data: {} });
  });

  it("a required field the file never mentions is missing", () => {
    const r = review("note: hi\n");
    expect(r.ignored).toBe(0);
    expect(r.missing).toBe(2);
    expect(r.issues.map((i) => i.field.join("."))).toEqual(["name", "data"]);
  });

  it("tells the two apart in the same file, and still lists every issue it found", () => {
    const r = review("name: 5\nnote: 5\ndata:\n  base_dir: d\n");
    expect(r.ignored).toBe(1); // note
    expect(r.missing).toBe(1); // name
    expect(r.issues.map((i) => i.field.join("."))).toEqual(["name", "note"]);
  });
});

/** The shape of the real classification schema where it matters here: a quoted
 *  number at the top of a section, numeric lists, a nullable list. */
const REAL_SHAPE: JsonSchema = {
  type: "object",
  properties: {
    name: { type: "string" },
    training: { $ref: "#/$defs/Training" },
    data: { $ref: "#/$defs/Data" },
    device: { $ref: "#/$defs/Device" },
    labels: { type: "array", items: { type: "string" } },
  },
  required: ["data"],
  $defs: {
    Training: {
      type: "object",
      properties: {
        learning_rate: { type: "number", exclusiveMinimum: 0 },
        epochs: { type: "integer", minimum: 1 },
        optimizer: { type: "string", enum: ["adam", "sgd"] },
        pretrained: { type: "boolean" },
      },
    },
    Data: {
      type: "object",
      properties: {
        base_dir: { type: "string" },
        transforms: { $ref: "#/$defs/Transforms" },
      },
      required: ["base_dir"],
    },
    Transforms: {
      type: "object",
      properties: {
        normalize_mean: { type: "array", items: { type: "number" } },
        normalize_std: { type: "array", items: { type: "number" } },
        rotation_degrees: { type: "integer" },
      },
    },
    Device: {
      type: "object",
      properties: {
        gpu_ids: { anyOf: [{ type: "array", items: { type: "integer" } }, { type: "null" }] },
      },
    },
  },
};

describe("a list with one bad item, as the real schema has them", () => {
  const review = (yaml: string) =>
    reviewImportedConfig(pt, parseYamlToConfig(yaml), REAL_SHAPE, REAL_SHAPE.$defs);
  const transforms = (r: { data: Record<string, unknown> }) =>
    ((r.data.data as Record<string, unknown>).transforms ?? {}) as Record<string, unknown>;

  it("takes normalize_std out whole instead of keeping two of three values", () => {
    const r = review(
      "data:\n  base_dir: d\n  transforms:\n    normalize_std: [0.229, oops, 0.225]\n    rotation_degrees: 5\n",
    );
    expect("normalize_std" in transforms(r)).toBe(false);
    expect(transforms(r).rotation_degrees).toBe(5);
    expect(r.ignored).toBe(1);
    expect(r.missing).toBe(0);
    // The user is still told which item it was.
    expect(r.issues.map((i) => i.field.join("."))).toEqual([
      "data.transforms.normalize_std.1",
    ]);
  });

  it("takes normalize_mean out whole when one item is not a number", () => {
    const r = review(
      "data:\n  base_dir: d\n  transforms:\n    normalize_mean: ['0.485', none, '0.406']\n",
    );
    expect("normalize_mean" in transforms(r)).toBe(false);
    expect(r.ignored).toBe(1);
  });

  it("takes gpu_ids out whole instead of silently losing a GPU", () => {
    const r = review("data:\n  base_dir: d\ndevice:\n  gpu_ids: [0, one]\n");
    expect(r.data.device).toEqual({});
    expect(r.ignored).toBe(1);
  });

  it("counts each dropped list once, however many of its items were bad", () => {
    const r = review(
      "data:\n  base_dir: d\n  transforms:\n    normalize_mean: [a, b, c]\n    normalize_std: [0.2, x, y]\ndevice:\n  gpu_ids: [z]\n",
    );
    expect(r.issues).toHaveLength(6);
    expect(r.ignored).toBe(3);
    expect(r.missing).toBe(0);
  });

  it("counts a required list that had to go as missing, not as ignored", () => {
    const strict: JsonSchema = {
      type: "object",
      properties: { labels: { type: "array", items: { type: "string" } } },
      required: ["labels"],
    };
    const r = reviewImportedConfig(pt, { labels: ["a", 5] }, strict, {});
    expect(r.data).toEqual({});
    expect(r.ignored).toBe(0);
    expect(r.missing).toBe(1);
  });
});

describe("coerceNumericStrings", () => {
  const coerce = (data: Record<string, unknown>) =>
    coerceNumericStrings(data, REAL_SHAPE, REAL_SHAPE.$defs);
  const training = (data: Record<string, unknown>) =>
    coerce({ training: data }).training as Record<string, unknown>;

  it("reads a quoted number the way the backend does", () => {
    expect(training({ learning_rate: "0.001" }).learning_rate).toBe(0.001);
    expect(training({ epochs: "10" }).epochs).toBe(10);
    expect(training({ learning_rate: "1e-4" }).learning_rate).toBe(0.0001);
    expect(training({ learning_rate: " -0.5 " }).learning_rate).toBe(-0.5);
    expect(training({ learning_rate: ".5" }).learning_rate).toBe(0.5);
    expect(training({ epochs: "+5" }).epochs).toBe(5);
  });

  it("takes an integer written with a zero fraction, but not a real fraction", () => {
    expect(training({ epochs: "10.0" }).epochs).toBe(10);
    expect(training({ epochs: "10.5" }).epochs).toBe("10.5");
  });

  it("leaves anything that is not plainly a finite number alone", () => {
    for (const text of ["abc", "", "   ", "0x10", "NaN", "Infinity", "-inf", "1,5", "1_000", "1e", "5 6"]) {
      expect(training({ learning_rate: text }).learning_rate).toBe(text);
    }
  });

  it("only touches what the schema says is a number", () => {
    const out = coerce({
      name: "123",
      training: { optimizer: "5", pretrained: "1", epochs: 3, learning_rate: null },
    });
    expect(out.name).toBe("123");
    expect(out.training).toEqual({ optimizer: "5", pretrained: "1", epochs: 3, learning_rate: null });
  });

  it("goes into the items of a list, and through a nullable list", () => {
    const out = coerce({
      data: { transforms: { normalize_mean: ["0.485", "0.456", "0.406"], normalize_std: [0.2, "0.3"] } },
      device: { gpu_ids: [0, "1"] },
      labels: ["1", "2"],
    });
    expect(out.data).toEqual({
      transforms: { normalize_mean: [0.485, 0.456, 0.406], normalize_std: [0.2, 0.3] },
    });
    expect(out.device).toEqual({ gpu_ids: [0, 1] });
    // A list of strings is still a list of strings.
    expect(out.labels).toEqual(["1", "2"]);
  });

  it("keeps a list item that is not numeric where it is, for the validator to flag", () => {
    const out = coerce({ data: { transforms: { normalize_std: [0.2, "oops", "0.4"] } } });
    expect(out.data).toEqual({ transforms: { normalize_std: [0.2, "oops", 0.4] } });
  });

  it("does not read a string where the field is also allowed to be one, or a fixed set", () => {
    const schema: JsonSchema = {
      type: "object",
      properties: {
        either: { anyOf: [{ type: "integer" }, { type: "string" }] },
        level: { type: "integer", enum: [1, 2] },
      },
    };
    const data = { either: "5", level: "1" };
    expect(coerceNumericStrings(data, schema, {})).toBe(data);
  });

  it("reads the values of a dict whose schema says they are numbers", () => {
    const schema: JsonSchema = {
      type: "object",
      properties: { weights: { type: "object", additionalProperties: { type: "number" } } },
    };
    expect(coerceNumericStrings({ weights: { a: "1.5", b: 2 } }, schema, {})).toEqual({
      weights: { a: 1.5, b: 2 },
    });
  });

  it("returns the same object when nothing needed reading, and never edits its input", () => {
    const clean = { training: { learning_rate: 0.01 }, name: "x" };
    expect(coerce(clean)).toBe(clean);
    const quoted = { training: { learning_rate: "0.01" }, name: "x" };
    const snapshot = structuredClone(quoted);
    const out = coerce(quoted);
    expect(quoted).toEqual(snapshot);
    expect(out).not.toBe(quoted);
    expect(out.name).toBe("x");
  });
});

describe("a quoted number in an imported file", () => {
  const review = (yaml: string) =>
    reviewImportedConfig(pt, parseYamlToConfig(yaml), REAL_SHAPE, REAL_SHAPE.$defs);
  const training = (r: { data: Record<string, unknown> }) =>
    (r.data.training ?? {}) as Record<string, unknown>;

  it("keeps lr: '0.001' as 0.001 instead of swapping in the default", () => {
    const r = review("data:\n  base_dir: d\ntraining:\n  learning_rate: '0.001'\n");
    expect(training(r).learning_rate).toBe(0.001);
    expect(r.issues).toEqual([]);
    expect(r.ignored).toBe(0);
  });

  it("keeps epochs: '10' as 10", () => {
    const r = review("data:\n  base_dir: d\ntraining:\n  epochs: '10'\n");
    expect(training(r).epochs).toBe(10);
    expect(r.issues).toEqual([]);
  });

  it("still flags epochs: '10.5', which the backend refuses too", () => {
    const r = review("data:\n  base_dir: d\ntraining:\n  epochs: '10.5'\n");
    expect("epochs" in training(r)).toBe(false);
    expect(r.issues.map((i) => i.field.join("."))).toEqual(["training.epochs"]);
    expect(r.ignored).toBe(1);
  });

  it("still flags a string that is not a number, and a boolean written as text", () => {
    const r = review(
      "data:\n  base_dir: d\ntraining:\n  learning_rate: fast\n  epochs: ten\n  pretrained: 'yes'\n",
    );
    expect(r.issues.map((i) => i.field.join("."))).toEqual([
      "training.learning_rate",
      "training.epochs",
      "training.pretrained",
    ]);
    expect(r.ignored).toBe(3);
  });

  it("reads quoted numbers in lists, so normalize_mean and gpu_ids load whole", () => {
    const r = review(
      "data:\n  base_dir: d\n  transforms:\n    normalize_mean: ['0.485', '0.456', '0.406']\ndevice:\n  gpu_ids: [0, '1']\n",
    );
    expect(r.issues).toEqual([]);
    expect(r.data.device).toEqual({ gpu_ids: [0, 1] });
    expect((r.data.data as Record<string, unknown>).transforms).toEqual({
      normalize_mean: [0.485, 0.456, 0.406],
    });
  });
});

describe("checkImportedConfig (the panels that refuse a file with problems)", () => {
  const check = (t: typeof pt, yaml: string) =>
    checkImportedConfig(t, parseYamlToConfig(yaml), REAL_SHAPE, REAL_SHAPE.$defs);

  it("lets a file with quoted numbers through, carrying them as numbers", () => {
    // The panels build their form from this data: a string in a number field
    // would silently fall back to the default there.
    const r = check(pt, "data:\n  base_dir: d\ntraining:\n  epochs: '5'\n  learning_rate: '0.02'\n");
    expect(r).toEqual({
      data: { data: { base_dir: "d" }, training: { epochs: 5, learning_rate: 0.02 } },
    });
  });

  it("refuses what the backend refuses, naming the field, in the language it is given", () => {
    const yaml = "data:\n  base_dir: d\ntraining:\n  epochs: '10.5'\n";
    expect(check(pt, yaml)).toEqual({ problem: "training › epochs: Esperado um inteiro." });
    expect(check(en, yaml)).toEqual({ problem: "training › epochs: Expected an integer." });
  });

  it("shows the first five problems and no more", () => {
    const r = check(
      pt,
      "training:\n  learning_rate: a\n  epochs: b\n  optimizer: 1\n  pretrained: c\ndata:\n  base_dir: 1\n  transforms:\n    rotation_degrees: x\n",
    );
    expect("problem" in r && r.problem.split("\n")).toHaveLength(5);
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
