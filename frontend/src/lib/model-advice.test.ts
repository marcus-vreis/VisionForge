import { describe, expect, it } from "vitest";
import type { ModelDefaults } from "../api/client";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { modelAdviceNote } from "./model-advice";

const base: ModelDefaults = {
  architecture: "resnet50",
  optimizer: "adam",
  learning_rate: 0.001,
  image_size: null,
  dataset_median_side: null,
  collapse_prone: false,
  note: null,
};

describe("modelAdviceNote", () => {
  const vgg: ModelDefaults = {
    ...base,
    architecture: "vgg16",
    learning_rate: 0.0001,
    collapse_prone: true,
    // The server's own Portuguese sentence; it must not be what is shown.
    note: "vgg16 com Adam a 1e-3 prevê uma classe só: medimos 0.25 de acurácia em 4 classes. Com adam a 0.0001 treina normal.",
  };

  it("warns about a collapse-prone architecture, in the language of the interface", () => {
    expect(modelAdviceNote(en, vgg)).toBe(
      "vgg16 with Adam at 1e-3 predicts a single class: we measured 0.25 accuracy on 4 classes. With adam at 0.0001 it trains normally.",
    );
  });

  it("says in Portuguese what the server says today", () => {
    expect(modelAdviceNote(pt, vgg)).toBe(vgg.note);
    expect(modelAdviceNote(pt, { ...vgg, architecture: "swin_t", optimizer: "adamw" })).toBe(
      "swin_t com Adam a 1e-3 prevê uma classe só: medimos 0.25 de acurácia em 4 classes. Com adamw a 0.0001 treina normal.",
    );
  });

  it("warns about upscaling when the images are smaller than the suggested size", () => {
    const small: ModelDefaults = { ...base, image_size: 224, dataset_median_side: 32 };
    expect(modelAdviceNote(pt, small)).toBe(
      "As imagens têm cerca de 32px de lado; treinar acima disso amplia a imagem sem acrescentar detalhe.",
    );
    expect(modelAdviceNote(en, small)).toBe(
      "The images are about 32px on a side; training above that upsizes the image without adding detail.",
    );
  });

  it("puts the collapse warning ahead of the upscaling one", () => {
    expect(modelAdviceNote(en, { ...vgg, image_size: 224, dataset_median_side: 32 })).toMatch(
      /^vgg16 with Adam at 1e-3/,
    );
  });

  it("stays quiet about upscaling when the images are as large as the size", () => {
    expect(modelAdviceNote(en, { ...base, image_size: 224, dataset_median_side: 224 })).toBe(
      "For resnet50, the measured setting is adam at 0.001.",
    );
    expect(modelAdviceNote(en, { ...base, image_size: null, dataset_median_side: 32 })).toBe(
      "For resnet50, the measured setting is adam at 0.001.",
    );
  });
});
