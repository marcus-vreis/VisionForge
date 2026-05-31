import { describe, expect, it } from "vitest";
import type { JsonSchema } from "../types/schema";
import {
  coerceGridValue,
  isGridableField,
  suggestNextGridValue,
  validateGridValue,
} from "./grid-axis";

const lrSchema: JsonSchema = { type: "number", exclusiveMinimum: 0 };
const intSchema: JsonSchema = { type: "integer", minimum: 1 };
const enumSchema: JsonSchema = { enum: ["adam", "sgd", "adamw"] };

describe("isGridableField", () => {
  it("accepts number and enum controls", () => {
    expect(isGridableField("learning_rate", "number")).toBe(true);
    expect(isGridableField("optimizer", "segmented")).toBe(true);
    expect(isGridableField("name", "select")).toBe(true);
  });

  it("rejects non-scalar controls", () => {
    expect(isGridableField("pretrained", "toggle")).toBe(false);
    expect(isGridableField("normalize_mean", "array-number")).toBe(false);
  });

  it("rejects excluded fields even when the control qualifies", () => {
    expect(isGridableField("num_classes", "number")).toBe(false);
    expect(isGridableField("seed", "number")).toBe(false);
    expect(isGridableField("task", "select")).toBe(false);
  });
});

describe("coerceGridValue", () => {
  it("parses numeric text to number", () => {
    expect(coerceGridValue("0.001", false)).toBe(0.001);
    expect(coerceGridValue(" 32 ", false)).toBe(32);
  });

  it("keeps non-numeric text as string for inline validation", () => {
    expect(coerceGridValue("abc", false)).toBe("abc");
  });

  it("never coerces enum values", () => {
    expect(coerceGridValue("adam", true)).toBe("adam");
  });
});

describe("validateGridValue", () => {
  it("flags learning_rate <= 0 (exclusiveMinimum)", () => {
    expect(validateGridValue("learning_rate", lrSchema, 0)).toMatch(/> 0/);
    expect(validateGridValue("learning_rate", lrSchema, 0.001)).toBeNull();
  });

  it("requires integers for integer fields", () => {
    expect(validateGridValue("epochs", intSchema, 2.5)).toMatch(/inteiro/);
    expect(validateGridValue("epochs", intSchema, 10)).toBeNull();
  });

  it("enforces minimum", () => {
    expect(validateGridValue("epochs", intSchema, 0)).toMatch(/≥ 1/);
  });

  it("enforces power-of-two for batch_size", () => {
    expect(validateGridValue("batch_size", intSchema, 24)).toMatch(/potência/);
    expect(validateGridValue("batch_size", intSchema, 32)).toBeNull();
  });

  it("restricts enums to declared options", () => {
    expect(validateGridValue("optimizer", enumSchema, "rmsprop")).toMatch(
      /fora das opções/,
    );
    expect(validateGridValue("optimizer", enumSchema, "adam")).toBeNull();
  });

  it("flags empty / NaN entries", () => {
    expect(validateGridValue("learning_rate", lrSchema, "")).toMatch(/inválido/);
  });
});

describe("suggestNextGridValue", () => {
  it("doubles batch_size to stay on powers of two", () => {
    expect(suggestNextGridValue(16, "batch_size", intSchema)).toBe(32);
  });

  it("halves a free float and increments an integer", () => {
    expect(suggestNextGridValue(0.001, "learning_rate", lrSchema)).toBe(0.0005);
    expect(suggestNextGridValue(10, "epochs", intSchema)).toBe(11);
  });

  it("picks a different enum option", () => {
    expect(suggestNextGridValue("adam", "optimizer", enumSchema)).toBe("sgd");
  });
});
