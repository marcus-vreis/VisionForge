import { beforeEach, describe, expect, it } from "vitest";
import {
  datasetLabel,
  isAutoName,
  rememberSuggestion,
  resetSuggestions,
  shortKey,
  suggestCustomTaskName,
  suggestExperimentName,
} from "./experiment-name";

describe("datasetLabel", () => {
  it("takes the folder name, on Windows and POSIX paths", () => {
    expect(datasetLabel("C:\\data\\CIFAR-10")).toBe("cifar-10");
    expect(datasetLabel("/home/m/datasets/oxford_pet/")).toBe("oxford_pet");
  });

  it("names a data.yaml by its stem, or by its folder when the stem is generic", () => {
    expect(datasetLabel("D:/voc/voc2007.yaml")).toBe("voc2007");
    expect(datasetLabel("D:/VOC2007/data.yaml")).toBe("voc2007");
  });

  it("skips folders that only say where the data sits", () => {
    expect(datasetLabel("D:/mvtec/bottle/train")).toBe("bottle");
  });

  it("is empty for an empty path", () => {
    expect(datasetLabel("  ")).toBe("");
  });
});

describe("suggestExperimentName", () => {
  it("joins dataset, model, epochs and learning rate", () => {
    expect(
      suggestExperimentName({
        datasetPath: "C:/data/cifar10",
        model: "resnet18",
        epochs: 20,
        learningRate: 0.001,
      }),
    ).toBe("cifar10_resnet18_e20_lr0.001");
  });

  it("writes small learning rates without exponent", () => {
    expect(
      suggestExperimentName({ datasetPath: "pet", learningRate: 1e-4 }),
    ).toBe("pet_lr0.0001");
  });

  it("names a grid by what it varies and leaves that value out of the base", () => {
    expect(
      suggestExperimentName({
        datasetPath: "cifar10",
        model: "resnet18",
        epochs: 20,
        learningRate: 0.001,
        strategy: "grid",
        sweepKeys: ["training.learning_rate", "training.batch_size"],
      }),
    ).toBe("cifar10_resnet18_e20_grid-lr-bs");
  });

  it("marks the other strategies", () => {
    const base = { datasetPath: "pet", model: "unet" };
    expect(suggestExperimentName({ ...base, strategy: "cv", folds: 5 })).toBe("pet_unet_cv5");
    expect(suggestExperimentName({ ...base, strategy: "replicates" })).toBe("pet_unet_rep");
    expect(suggestExperimentName({ ...base, strategy: "simple" })).toBe("pet_unet");
  });

  it("falls back to a placeholder dataset when none is chosen", () => {
    expect(suggestExperimentName({ datasetPath: "", model: "yolo11n" })).toBe(
      "experimento_yolo11n",
    );
  });
});

describe("shortKey", () => {
  it("matches the backend's short keys", () => {
    expect(shortKey("training.lr0")).toBe("lr");
    expect(shortKey("training.weight_decay")).toBe("wd");
    expect(shortKey("training.scheduler.kind")).toBe("kind");
  });
});

describe("isAutoName", () => {
  beforeEach(() => resetSuggestions());

  it("replaces empty names, task defaults and the previous suggestion only", () => {
    expect(isAutoName("detection", "", ["detection_001"])).toBe(true);
    expect(isAutoName("detection", "detection_001", ["detection_001"])).toBe(true);
    rememberSuggestion("detection", "voc_yolo11n_e50");
    expect(isAutoName("detection", "voc_yolo11n_e50", [])).toBe(true);
    expect(isAutoName("detection", "meu_teste", ["detection_001"])).toBe(false);
  });

  it("keeps suggestions apart per task", () => {
    rememberSuggestion("detection", "a");
    expect(isAutoName("segmentation", "a", [])).toBe(false);
  });
});

describe("suggestCustomTaskName", () => {
  const training = { epochs: 10, learning_rate: 0.001 };

  it("names a task with no dataset folder after the task", () => {
    expect(suggestCustomTaskName("example_counting", { data: { base_dir: "" }, training }, "simple")).toBe(
      "example_counting_e10_lr0.001",
    );
  });

  it("prefers the dataset folder when there is one", () => {
    expect(
      suggestCustomTaskName("example_counting", { data: { base_dir: "C:/data/contagem" }, training }, "simple"),
    ).toBe("contagem_e10_lr0.001");
  });

  it("includes the model when the task's config has one", () => {
    expect(
      suggestCustomTaskName("t", { data: { base_dir: "pet" }, model: { name: "unet" }, training }, "simple"),
    ).toBe("pet_unet_e10_lr0.001");
  });
});
