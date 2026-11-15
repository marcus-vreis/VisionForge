import { describe, expect, it } from "vitest";

import { alternateModel } from "./model-source";

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
