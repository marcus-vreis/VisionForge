import { describe, expect, it } from "vitest";

import { buildComparePayload, toggleModel } from "./compare-form";

const order = ["autoencoder", "patchcore", "padim"];

describe("toggleModel", () => {
  it("removes a selected model", () => {
    expect(toggleModel(["autoencoder", "patchcore"], "autoencoder", order)).toEqual([
      "patchcore",
    ]);
  });

  it("adds a model back in the order the options are listed", () => {
    expect(toggleModel(["padim"], "autoencoder", order)).toEqual(["autoencoder", "padim"]);
  });
});

describe("buildComparePayload", () => {
  it("is the body /{task}/compare reads next to the config", () => {
    const names = ["autoencoder", "patchcore"];
    const payload = buildComparePayload(names, "auroc");
    expect(payload).toEqual({ model_names: ["autoencoder", "patchcore"], metric: "auroc" });
    // A copy: later toggles in the card do not change a submitted request.
    expect(payload.model_names).not.toBe(names);
  });
});
