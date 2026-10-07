import { describe, expect, it } from "vitest";
import type { CollapseEvidence, ModelDefaults } from "../api/client";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { isAlarming, modelAdviceNote } from "./model-advice";

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

// No recovery run on this setup (VGG16 at 1e-4 was only run on two classes).
const vggEvidence: CollapseEvidence = {
  measured_on: "vgg16",
  accuracy: 0.25,
  outcome: "collapse",
  recovered_accuracy: null,
};
const swinEvidence: CollapseEvidence = {
  measured_on: "swin_t",
  accuracy: 0.25,
  outcome: "collapse",
  recovered_accuracy: 0.88,
};
const vitEvidence: CollapseEvidence = {
  measured_on: "vit_b_16",
  accuracy: 0.41,
  outcome: "fails_to_learn",
  recovered_accuracy: 0.85,
};

describe("modelAdviceNote", () => {
  describe("a collapse that was measured, with no recovery run", () => {
    const vgg16 = prone("vgg16", vggEvidence);

    it("cites the number and the model, and only suggests the rate, in English", () => {
      expect(modelAdviceNote(en, vgg16)).toBe(
        "vgg16 predicted a single class with Adam at 1e-3 (accuracy 0.25 on 4 classes). We suggest adam at 0.0001.",
      );
    });

    it("cites the number and the model, and only suggests the rate, in Portuguese", () => {
      expect(modelAdviceNote(pt, vgg16)).toBe(
        "vgg16 previu uma classe só com Adam a 1e-3 (acurácia 0.25 em 4 classes). Sugerimos adam a 0.0001.",
      );
    });

    it("does not promise the new setting trains", () => {
      expect(modelAdviceNote(en, vgg16)).not.toMatch(/trains normally/);
      expect(modelAdviceNote(pt, vgg16)).not.toMatch(/treina normal/);
    });

    it("says whose number it is when the model is a sibling of the one measured", () => {
      const vgg19 = prone("vgg19", vggEvidence);
      expect(modelAdviceNote(en, vgg19)).toBe(
        "vgg19: vgg16, from the same family, predicted a single class with Adam at 1e-3 (accuracy 0.25 on 4 classes); this model was not measured. We suggest adam at 0.0001.",
      );
      expect(modelAdviceNote(pt, vgg19)).toBe(
        "vgg19: o vgg16, da mesma família, previu uma classe só com Adam a 1e-3 (acurácia 0.25 em 4 classes); este modelo não foi medido. Sugerimos adam a 0.0001.",
      );
    });

    it("prints the accuracy the response carries, to two places", () => {
      const other = prone("alexnet", {
        measured_on: "alexnet",
        accuracy: 0.3,
        outcome: "collapse",
        recovered_accuracy: null,
      });
      expect(modelAdviceNote(en, other)).toContain("accuracy 0.30 on 4 classes");
    });
  });

  describe("a collapse whose recovery was measured on the same setup", () => {
    const swin = prone("swin_t", swinEvidence, "adamw");

    it("states the recovery as a number, in both languages", () => {
      expect(modelAdviceNote(en, swin)).toBe(
        "swin_t predicted a single class with Adam at 1e-3 (accuracy 0.25 on 4 classes). With adamw at 0.0001, accuracy was 0.88 under the same conditions.",
      );
      expect(modelAdviceNote(pt, swin)).toBe(
        "swin_t previu uma classe só com Adam a 1e-3 (acurácia 0.25 em 4 classes). Com adamw a 0.0001, a acurácia foi 0.88 nas mesmas condições.",
      );
    });

    it("says in Portuguese what the server says", () => {
      // Parity with src/visionforge/gui/api/model_notes.py: same sentence either way.
      expect(modelAdviceNote(pt, swin)).toBe(
        "swin_t previu uma classe só com Adam a 1e-3 (acurácia 0.25 em 4 classes). Com adamw a 0.0001, a acurácia foi 0.88 nas mesmas condições.",
      );
    });

    it("never gives a sibling the recovery of the model that was run", () => {
      const swinB = prone("swin_b", swinEvidence, "adamw");
      expect(modelAdviceNote(en, swinB)).toBe(
        "swin_b: swin_t, from the same family, predicted a single class with Adam at 1e-3 (accuracy 0.25 on 4 classes); this model was not measured. We suggest adamw at 0.0001.",
      );
      expect(modelAdviceNote(pt, swinB)).not.toContain("0.88");
    });

    it("leaves the recovery out when the response omits it", () => {
      const noRecovery = prone("swin_t", { ...swinEvidence, recovered_accuracy: undefined });
      expect(modelAdviceNote(en, noRecovery)).toMatch(/We suggest adam at 0\.0001\.$/);
      expect(modelAdviceNote(en, noRecovery)).not.toContain("0.88");
    });
  });

  describe("a model that failed to learn without collapsing", () => {
    const vit = prone("vit_b_16", vitEvidence, "adamw");

    it("does not say it predicted one class, in either language", () => {
      expect(modelAdviceNote(en, vit)).toBe(
        "vit_b_16 did not learn with Adam at 1e-3 (accuracy 0.41 on 4 classes). With adamw at 0.0001, accuracy was 0.85 under the same conditions.",
      );
      expect(modelAdviceNote(pt, vit)).toBe(
        "vit_b_16 não aprendeu com Adam a 1e-3 (acurácia 0.41 em 4 classes). Com adamw a 0.0001, a acurácia foi 0.85 nas mesmas condições.",
      );
      expect(modelAdviceNote(en, vit)).not.toMatch(/single class/);
      expect(modelAdviceNote(pt, vit)).not.toMatch(/uma classe só/);
    });

    it("names the measured model for a sibling, without its recovery", () => {
      const vitL = prone("vit_l_16", vitEvidence, "adamw");
      expect(modelAdviceNote(en, vitL)).toBe(
        "vit_l_16: vit_b_16, from the same family, did not learn with Adam at 1e-3 (accuracy 0.41 on 4 classes); this model was not measured. We suggest adamw at 0.0001.",
      );
      expect(modelAdviceNote(pt, vitL)).toBe(
        "vit_l_16: o vit_b_16, da mesma família, não aprendeu com Adam a 1e-3 (acurácia 0.41 em 4 classes); este modelo não foi medido. Sugerimos adamw a 0.0001.",
      );
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

  describe("an architecture that was never flagged", () => {
    it("suggests the setting instead of calling it measured, in both languages", () => {
      // resnet18, mobilenet and the rest were never in the measured grid, and
      // the note is shown for any of them whose form differs from the suggestion.
      for (const architecture of ["resnet50", "resnet18", "mobilenet_v3_small"]) {
        const advice = { ...base, architecture };
        expect(modelAdviceNote(en, advice)).toBe(`For ${architecture}, we suggest adam at 0.001.`);
        expect(modelAdviceNote(pt, advice)).toBe(`Para ${architecture}, sugerimos adam a 0.001.`);
      }
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
      "For resnet50, we suggest adam at 0.001.",
    );
    expect(modelAdviceNote(en, { ...base, image_size: null, dataset_median_side: 32 })).toBe(
      "For resnet50, we suggest adam at 0.001.",
    );
  });
});

describe("isAlarming", () => {
  it("is true where a failure was measured", () => {
    expect(isAlarming(prone("vgg16", vggEvidence))).toBe(true);
    expect(isAlarming(prone("vit_l_16", vitEvidence))).toBe(true);
  });

  it("is false for a flagged family with no evidence: the suggestion stays, the alarm does not", () => {
    expect(isAlarming(prone("maxvit_t", null, "adamw"))).toBe(false);
  });

  it("is false for a response that predates collapse_evidence", () => {
    const old: ModelDefaults = prone("vgg16", null);
    delete old.collapse_evidence;
    expect(isAlarming(old)).toBe(false);
  });

  it("is false for an outcome this build does not know", () => {
    const future = prone("vgg16", {
      measured_on: "vgg16",
      accuracy: 0.25,
      outcome: "diverged" as CollapseEvidence["outcome"],
    });
    expect(isAlarming(future)).toBe(false);
  });

  it("is false when nothing was flagged", () => {
    expect(isAlarming(base)).toBe(false);
  });
});
