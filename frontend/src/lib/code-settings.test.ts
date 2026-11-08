import { describe, expect, it } from "vitest";
import { codeSettingsFor } from "./code-settings";

describe("codeSettingsFor", () => {
  it("reads what the payload decides", () => {
    expect(codeSettingsFor({
      task: "binary",
      training: { mixed_precision: false },
      transforms: { augment: true },
      model: { name: "resnet18" },
    })).toEqual({ kind: "binary", amp: false, augment: true, model: "resnet18" });
  });

  it("leaves mixed precision undecided unless it is off or the device is the CPU", () => {
    expect(codeSettingsFor({ training: { mixed_precision: true }, device: { kind: "auto" } }).amp).toBeUndefined();
    expect(codeSettingsFor({ training: { mixed_precision: true }, device: { kind: "cuda" } }).amp).toBeUndefined();
    expect(codeSettingsFor({ training: { mixed_precision: true }, device: { kind: "cpu" } }).amp).toBe(false);
  });

  it("ignores values it does not know", () => {
    expect(codeSettingsFor({ task: "regression", model: { name: "" }, transforms: [] })).toEqual({});
  });
});
