import { describe, expect, it } from "vitest";

import {
  BUILTIN_MODEL,
  alternateModel,
  customModelChoice,
  customModelOptions,
  withCustomModel,
} from "./model-source";

describe("alternateModel", () => {
  it("is null for the builtin model", () => {
    expect(alternateModel({ name: "resnet18", pretrained: true })).toBeNull();
    expect(alternateModel({})).toBeNull();
  });

  it("names the researcher's own model", () => {
    expect(alternateModel({ name: "resnet18", custom_model: "my_net" })).toEqual({
      source: "custom_model",
      name: "my_net",
    });
  });

  it("names a timm architecture", () => {
    expect(alternateModel({ timm_model: " convnext_tiny " })).toEqual({
      source: "timm_model",
      name: "convnext_tiny",
    });
  });

  it("reads an empty or blank name as not set, like the backend", () => {
    expect(alternateModel({ custom_model: "", timm_model: "   " })).toBeNull();
    expect(alternateModel({ custom_model: null, timm_model: 3 })).toBeNull();
  });
});

describe("the custom-model selector", () => {
  const words = { builtin: "None", builtinSub: "builtin", missing: "missing" };

  it("shows the builtin choice when no custom model is set", () => {
    expect(customModelChoice({ name: "resnet18" })).toBe(BUILTIN_MODEL);
    expect(customModelChoice({ custom_model: null })).toBe(BUILTIN_MODEL);
    expect(customModelChoice({ custom_model: "  " })).toBe(BUILTIN_MODEL);
  });

  it("shows the custom model the form holds, trimmed", () => {
    expect(customModelChoice({ custom_model: " my_net " })).toBe("my_net");
  });

  it("sets custom_model and clears timm_model, which the config refuses together", () => {
    expect(withCustomModel({ name: "resnet18", timm_model: "convnext_tiny" }, "my_net")).toEqual({
      name: "resnet18",
      custom_model: "my_net",
      timm_model: null,
    });
  });

  it("clears custom_model when the builtin architecture is picked", () => {
    expect(withCustomModel({ name: "resnet18", custom_model: "my_net" }, BUILTIN_MODEL)).toEqual({
      name: "resnet18",
      custom_model: null,
    });
  });

  it("offers the builtin architecture first, then each registered model", () => {
    expect(customModelOptions(["a_net", "b_net"], BUILTIN_MODEL, words)).toEqual([
      { value: BUILTIN_MODEL, label: "None", sub: "builtin" },
      { value: "a_net", label: "a_net" },
      { value: "b_net", label: "b_net" },
    ]);
  });

  it("keeps a chosen name the folder does not register, marked missing", () => {
    expect(customModelOptions(["a_net"], "gone_net", words)).toEqual([
      { value: BUILTIN_MODEL, label: "None", sub: "builtin" },
      { value: "a_net", label: "a_net" },
      { value: "gone_net", label: "gone_net", sub: "missing" },
    ]);
  });
});
