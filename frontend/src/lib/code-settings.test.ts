import { describe, expect, it } from "vitest";
import { codeSettingsFor } from "./code-settings";

describe("codeSettingsFor", () => {
  it("reads what the payload decides", () => {
    expect(codeSettingsFor("classification", {
      task: "binary",
      training: { mixed_precision: false },
      transforms: { augment: true },
      model: { name: "resnet18" },
    })).toEqual({ kind: "binary", amp: false, augment: true, model: "resnet18" });
  });

  it("leaves mixed precision undecided unless it is off or the device is the CPU", () => {
    const amp = (device: string) =>
      codeSettingsFor("classification", { training: { mixed_precision: true }, device: { kind: device } }).amp;
    expect(amp("auto")).toBeUndefined();
    expect(amp("cuda")).toBeUndefined();
    expect(amp("cpu")).toBe(false);
  });

  it("ignores values it does not know", () => {
    expect(codeSettingsFor("classification", { task: "regression", model: { name: "" }, transforms: [] })).toEqual({});
  });

  it("reads the loss of a regression run", () => {
    expect(codeSettingsFor("regression", {
      task: "regression",
      model: { name: "resnet18" },
      training: { loss: "huber", mixed_precision: false },
    })).toEqual({ loss: "huber" });
    expect(codeSettingsFor("regression", { training: {} })).toEqual({});
  });

  it("decides nothing for a task without settings", () => {
    expect(codeSettingsFor("custom", { training: { loss: "mse" } })).toEqual({});
  });
});
