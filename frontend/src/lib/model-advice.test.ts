import { describe, expect, it } from "vitest";
import type { CollapseEvidence, ModelDefaults } from "../api/client";
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
  collapse_evidence: null,
  note: null,
};

/** A collapse-prone response, as the server sends it for `architecture`. */
function prone(
  architecture: string,
  evidence: CollapseEvidence | null,
  optimizer = "adam",
): ModelDefaults {
  return {
    ...base,
    architecture,
    optimizer,
    learning_rate: 0.0001,
    collapse_prone: true,
    collapse_evidence: evidence,
    // The server's own Portuguese sentence; it must not be what is shown.
    note: "texto do servidor",
  };
}

const vggEvidence: CollapseEvidence = { measured_on: "vgg16", accuracy: 0.25, outcome: "collapse" };
const vitEvidence: CollapseEvidence = {
  measured_on: "vit_b_16",
  accuracy: 0.41,
  outcome: "fails_to_learn",
};

describe("modelAdviceNote", () => {
  describe("a collapse that was measured", () => {
    const vgg16 = prone("vgg16", vggEvidence);

    it("cites the number and the model, in English", () => {
      expect(modelAdviceNote(en, vgg16)).toBe(
        "vgg16 predicted a single class with Adam at 1e-3 (accuracy 0.25 on 4 classes). With adam at 0.0001 it trains normally.",
      );
    });

    it("cites the number and the model, in Portuguese", () => {
      expect(modelAdviceNote(pt, vgg16)).toBe(
        "vgg16 previu uma classe só com Adam a 1e-3 (acurácia 0.25 em 4 classes). Com adam a 0.0001 treina normal.",
      );
    });

    it("says in Portuguese what the server says", () => {
      // Parity with src/visionforge/gui/api/model_notes.py: same sentence either way.
      expect(modelAdviceNote(pt, prone("swin_t", { ...vggEvidence, measured_on: "swin_t" }, "adamw"))).toBe(
        "swin_t previu uma classe só com Adam a 1e-3 (acurácia 0.25 em 4 classes). Com adamw a 0.0001 treina normal.",
      );
    });

    it("says whose number it is when the model is a sibling of the one measured", () => {
      const vgg19 = prone("vgg19", vggEvidence);
      expect(modelAdviceNote(en, vgg19)).toBe(
        "vgg19: vgg16, from the same family, predicted a single class with Adam at 1e-3 (accuracy 0.25 on 4 classes); this model was not measured. With adam at 0.0001 it trains normally.",
      );
      expect(modelAdviceNote(pt, vgg19)).toBe(
        "vgg19: o vgg16, da mesma família, previu uma classe só com Adam a 1e-3 (acurácia 0.25 em 4 classes); este modelo não foi medido. Com adam a 0.0001 treina normal.",
      );
    });

    it("prints the accuracy the response carries, to two places", () => {
      const other = prone("alexnet", { measured_on: "alexnet", accuracy: 0.3, outcome: "collapse" });
      expect(modelAdviceNote(en, other)).toContain("accuracy 0.30 on 4 classes");
    });
  });

  describe("a model that failed to learn without collapsing", () => {
    const vit = prone("vit_b_16", vitEvidence, "adamw");

    it("does not say it predicted one class, in either language", () => {
      expect(modelAdviceNote(en, vit)).toBe(
        "vit_b_16 did not learn with Adam at 1e-3 (accuracy 0.41 on 4 classes). With adamw at 0.0001 it trains normally.",
      );
      expect(modelAdviceNote(pt, vit)).toBe(
        "vit_b_16 não aprendeu com Adam a 1e-3 (acurácia 0.41 em 4 classes). Com adamw a 0.0001 treina normal.",
      );
      expect(modelAdviceNote(en, vit)).not.toMatch(/single class/);
      expect(modelAdviceNote(pt, vit)).not.toMatch(/uma classe só/);
    });

    it("names the measured model for a sibling", () => {
      const vitL = prone("vit_l_16", vitEvidence, "adamw");
      expect(modelAdviceNote(en, vitL)).toContain("vit_l_16: vit_b_16, from the same family, did not learn");
      expect(modelAdviceNote(pt, vitL)).toContain("vit_l_16: o vit_b_16, da mesma família, não aprendeu");
      expect(modelAdviceNote(en, vitL)).toContain("this model was not measured");
    });
  });

  describe("a family that was never measured", () => {
    const maxvit = prone("maxvit_t", null, "adamw");

    it("quotes no accuracy, in either language", () => {
      expect(modelAdviceNote(en, maxvit)).toBe(
        "maxvit_t: Adam at 1e-3 was not measured for this family; we suggest adamw at 0.0001, the same as the measured attention families.",
      );
      expect(modelAdviceNote(pt, maxvit)).toBe(
        "maxvit_t: Adam a 1e-3 não foi medido para esta família; sugerimos adamw a 0.0001, o mesmo das famílias de atenção medidas.",
      );
    });
  });

  describe("a response from a server that predates collapse_evidence", () => {
    // The field is simply absent; the flag alone must not bring a number back.
    const old: ModelDefaults = prone("vgg16", null);
    delete old.collapse_evidence;

    it("falls back to the unmeasured wording", () => {
      expect(modelAdviceNote(en, old)).toBe(
        "vgg16: Adam at 1e-3 was not measured for this family; we suggest adam at 0.0001, the same as the measured attention families.",
      );
      expect(modelAdviceNote(pt, old)).toBe(
        "vgg16: Adam a 1e-3 não foi medido para esta família; sugerimos adam a 0.0001, o mesmo das famílias de atenção medidas.",
      );
    });

    it("never prints an accuracy the response did not carry", () => {
      expect(modelAdviceNote(en, old)).not.toMatch(/0\.25|0\.41|accuracy/);
      expect(modelAdviceNote(pt, old)).not.toMatch(/0\.25|0\.41|acurácia/);
    });

    it("does the same for an outcome this build does not know", () => {
      const future = prone("vgg16", {
        measured_on: "vgg16",
        accuracy: 0.25,
        outcome: "diverged" as CollapseEvidence["outcome"],
      });
      expect(modelAdviceNote(en, future)).toMatch(/was not measured for this family/);
    });
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
    const vgg16 = { ...prone("vgg16", vggEvidence), image_size: 224, dataset_median_side: 32 };
    expect(modelAdviceNote(en, vgg16)).toMatch(/^vgg16 predicted a single class with Adam at 1e-3/);
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
