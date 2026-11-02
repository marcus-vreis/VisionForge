import { describe, expect, it } from "vitest";
import type { TrialLearningRate } from "../api/client";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { collapseWarning, learningRatePlanRequest, rateIsPerModel } from "./lr-plan";

const flagged: TrialLearningRate = {
  architecture: "vgg16",
  optimizer: "adam",
  learning_rate: 0.0003,
  multiplier: 3,
  collapse_prone: true,
  collapse_evidence: { measured_on: "vgg16", accuracy: 0.25, outcome: "collapse" },
};

describe("learningRatePlanRequest", () => {
  const form = {
    block: "grid_search",
    model: { name: "resnet50" },
    training: { learning_rate: 0.001, optimizer: "adamw" },
    grid_search: {
      hyperparameters: { "model.name": ["vgg16"], "training.learning_rate": [3] },
      lr_mode: "suggested_multiplier",
    },
  };

  it("carries what decides each rate", () => {
    expect(learningRatePlanRequest(form)).toEqual({
      block: "grid_search",
      model_name: "resnet50",
      optimizer: "adamw",
      learning_rate: 0.001,
      hyperparameters: { "model.name": ["vgg16"], "training.learning_rate": [3] },
      lr_mode: "suggested_multiplier",
      model_names: [],
      lr_policy: "fixed",
    });
  });

  it("waits while the rate is not a number", () => {
    expect(learningRatePlanRequest({ training: { learning_rate: "" } })).toBeNull();
  });
});

describe("rateIsPerModel", () => {
  const training = { learning_rate: 0.001 };

  it("is true only where the form's rate is not the one used", () => {
    const grid = (lr_mode: string) => ({
      block: "grid_search",
      training,
      grid_search: { lr_mode },
    });
    const comparison = (lr_policy: string) => ({
      block: "model_comparison",
      training,
      model_comparison: { lr_policy },
    });
    expect(rateIsPerModel(grid("suggested_multiplier"))).toBe(true);
    expect(rateIsPerModel(grid("absolute"))).toBe(false);
    expect(rateIsPerModel(comparison("suggested"))).toBe(true);
    expect(rateIsPerModel(comparison("fixed"))).toBe(false);
    expect(rateIsPerModel({ block: "classification", training })).toBe(false);
  });
});

describe("collapseWarning", () => {
  it("says nothing about a combination that is not flagged", () => {
    expect(
      collapseWarning(pt, { ...flagged, collapse_prone: false, collapse_evidence: null }),
    ).toBeNull();
  });

  it("quotes the measurement and where it was taken", () => {
    const text = collapseWarning(en, flagged);
    expect(text).toContain("vgg16 with adam at 0.0003");
    expect(text).toContain("0.25");
    expect(text).toContain("Adam at 1e-3");
    expect(text).not.toContain("not measured");
  });

  it("keeps a sibling from borrowing the number", () => {
    const text = collapseWarning(pt, { ...flagged, architecture: "vgg19" });
    expect(text).toContain("vgg16");
    expect(text).toContain("vgg19 não foi medido");
  });

  it("quotes no number without evidence", () => {
    const text = collapseWarning(pt, { ...flagged, collapse_evidence: null });
    expect(text).not.toMatch(/0\.\d\d /);
    expect(text).toContain("não foi medida");
  });

  it("words the model that learned little apart from a collapse", () => {
    const text = collapseWarning(pt, {
      ...flagged,
      architecture: "vit_b_16",
      collapse_evidence: { measured_on: "vit_b_16", accuracy: 0.41, outcome: "fails_to_learn" },
    });
    expect(text).toContain("aprendeu pouco");
    expect(text).not.toContain("uma classe só");
  });
});
