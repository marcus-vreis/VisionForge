import { describe, expect, it } from "vitest";

import {
  buildCustomForm,
  buildCustomPayload,
  descriptorToDefinition,
  isCustomTask,
  mergeTasks,
  metricOptions,
  type TaskDescriptor,
} from "./custom-tasks";
import type { JsonSchema } from "../types/schema";
import { TASKS } from "../types/tasks";

function descriptor(over: Partial<TaskDescriptor> = {}): TaskDescriptor {
  return {
    key: "cell_counting",
    label: "Contagem de células",
    accent: "#2dd4bf",
    description: "Conte objetos",
    custom: true,
    metrics: { mae: "lower", rmse: "lower" },
    primary_metric: "mae",
    ...over,
  };
}

describe("mergeTasks", () => {
  it("keeps the five built-ins first and in order", () => {
    const merged = mergeTasks(TASKS, [descriptor()]);
    expect(merged.slice(0, TASKS.length).map((t) => t.key)).toEqual(
      TASKS.map((t) => t.key),
    );
    expect(merged).toHaveLength(TASKS.length + 1);
  });

  it("ignores built-in rows from the API (they are already local)", () => {
    const builtinRow = descriptor({ key: "classification", custom: false });
    expect(mergeTasks(TASKS, [builtinRow])).toHaveLength(TASKS.length);
  });

  it("never lets a custom task shadow a built-in key", () => {
    // Defence in depth: the backend rejects these keys, but a stale/hand-made
    // response must not replace a tab that always works.
    const shadow = descriptor({ key: "detection", custom: true });
    const merged = mergeTasks(TASKS, [shadow]);
    expect(merged).toHaveLength(TASKS.length);
    expect(merged.find((t) => t.key === "detection")?.label).toBe(
      TASKS.find((t) => t.key === "detection")?.label,
    );
  });

  it("sorts several customs by label", () => {
    const merged = mergeTasks(TASKS, [
      descriptor({ key: "zeta", label: "Zeta" }),
      descriptor({ key: "alpha", label: "Alpha" }),
    ]);
    expect(merged.slice(TASKS.length).map((t) => t.label)).toEqual([
      "Alpha",
      "Zeta",
    ]);
  });

  it("survives an empty task list", () => {
    expect(mergeTasks(TASKS, [])).toHaveLength(TASKS.length);
  });
});

describe("descriptorToDefinition", () => {
  it("carries identity and metric metadata, and no curated params", () => {
    const def = descriptorToDefinition(descriptor());
    expect(def.accent).toBe("#2dd4bf");
    expect(def.label).toBe("Contagem de células");
    expect(def.models).toEqual([]);
    expect(def.params).toEqual([]);
    expect(isCustomTask(def)).toBe(true);
    expect(def.primaryMetric).toBe("mae");
  });

  it("falls back to a description when the researcher left it empty", () => {
    expect(descriptorToDefinition(descriptor({ description: "" })).description)
      .toContain("pesquisador");
  });

  it("marks built-in definitions as not custom", () => {
    expect(isCustomTask(TASKS[0])).toBe(false);
  });
});

describe("buildCustomForm", () => {
  const schema: JsonSchema = {
    type: "object",
    properties: {
      name: { type: "string", default: "custom_run" },
      density_sigma: { type: "number", default: 2.0 },
      enabled: { type: "boolean" },
      training: { $ref: "#/$defs/Training" },
      optional: { anyOf: [{ type: "string" }, { type: "null" }] },
    },
    $defs: {
      Training: {
        type: "object",
        properties: {
          epochs: { type: "integer", default: 10 },
          optimizer: { enum: ["adam", "sgd"] },
        },
      },
    },
  };

  it("uses declared defaults and recurses through $ref", () => {
    const form = buildCustomForm(schema);
    expect(form.name).toBe("custom_run");
    expect(form.density_sigma).toBe(2.0);
    expect(form.training).toEqual({ epochs: 10, optimizer: "adam" });
  });

  it("falls back per JSON type when no default is declared", () => {
    const form = buildCustomForm(schema);
    expect(form.enabled).toBe(false);
    expect(form.optional).toBe("");
  });
});

describe("buildCustomPayload", () => {
  it("injects the live device selection over the form", () => {
    const payload = buildCustomPayload(
      { name: "run", device: { kind: "cpu", gpu_ids: null } },
      { kind: "cuda", gpu_ids: [0] },
    );
    expect(payload.device).toEqual({ kind: "cuda", gpu_ids: [0] });
    expect(payload.name).toBe("run");
  });
});

describe("metricOptions", () => {
  it("puts the primary metric first and shows each direction", () => {
    const def = descriptorToDefinition(
      descriptor({ metrics: { rmse: "lower", score: "higher" }, primary_metric: "score" }),
    );
    expect(metricOptions(def)).toEqual([
      { value: "score", label: "score ↑" },
      { value: "rmse", label: "rmse ↓" },
    ]);
  });
});
