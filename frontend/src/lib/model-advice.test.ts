import { describe, expect, it } from "vitest";
import type { CollapseEvidence, ModelDefaults } from "../api/client";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import type { Dict } from "../i18n/pt";
import { isAlarming, modelAdviceNote, type ModelAdviceTask } from "./model-advice";

/** The note as the classification form shows it, unless a task is named. */
const note = (t: Dict, advice: ModelDefaults, task: ModelAdviceTask = "classification") =>
  modelAdviceNote(t, advice, task);

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
      expect(note(en, vgg16)).toBe(
        "vgg16 predicted a single class with Adam at 1e-3 (accuracy 0.25 on 4 classes). We suggest adam at 0.0001.",
      );
    });

    it("cites the number and the model, and only suggests the rate, in Portuguese", () => {
      expect(note(pt, vgg16)).toBe(
        "vgg16 previu uma classe só com Adam a 1e-3 (acurácia 0.25 em 4 classes). Sugerimos adam a 0.0001.",
      );
    });

    it("does not promise the new setting trains", () => {
      expect(note(en, vgg16)).not.toMatch(/trains normally/);
      expect(note(pt, vgg16)).not.toMatch(/treina normal/);
    });

    it("says whose number it is when the model is a sibling of the one measured", () => {
      const vgg19 = prone("vgg19", vggEvidence);
      expect(note(en, vgg19)).toBe(
        "vgg19: vgg16, from the same family, predicted a single class with Adam at 1e-3 (accuracy 0.25 on 4 classes); this model was not measured. We suggest adam at 0.0001.",
      );
      expect(note(pt, vgg19)).toBe(
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
      expect(note(en, other)).toContain("accuracy 0.30 on 4 classes");
    });
  });

  describe("a collapse whose recovery was measured on the same setup", () => {
    const swin = prone("swin_t", swinEvidence, "adamw");

    it("states the recovery as a number, in both languages", () => {
      expect(note(en, swin)).toBe(
        "swin_t predicted a single class with Adam at 1e-3 (accuracy 0.25 on 4 classes). With adamw at 0.0001, accuracy was 0.88 under the same conditions.",
      );
      expect(note(pt, swin)).toBe(
        "swin_t previu uma classe só com Adam a 1e-3 (acurácia 0.25 em 4 classes). Com adamw a 0.0001, a acurácia foi 0.88 nas mesmas condições.",
      );
    });

    it("says in Portuguese what the server says", () => {
      // Parity with src/visionforge/gui/api/model_notes.py: same sentence either way.
      expect(note(pt, swin)).toBe(
        "swin_t previu uma classe só com Adam a 1e-3 (acurácia 0.25 em 4 classes). Com adamw a 0.0001, a acurácia foi 0.88 nas mesmas condições.",
      );
    });

    it("never gives a sibling the recovery of the model that was run", () => {
      const swinB = prone("swin_b", swinEvidence, "adamw");
      expect(note(en, swinB)).toBe(
        "swin_b: swin_t, from the same family, predicted a single class with Adam at 1e-3 (accuracy 0.25 on 4 classes); this model was not measured. We suggest adamw at 0.0001.",
      );
      expect(note(pt, swinB)).not.toContain("0.88");
    });

    it("leaves the recovery out when the response omits it", () => {
      const noRecovery = prone("swin_t", { ...swinEvidence, recovered_accuracy: undefined });
      expect(note(en, noRecovery)).toMatch(/We suggest adam at 0\.0001\.$/);
      expect(note(en, noRecovery)).not.toContain("0.88");
    });
  });

  describe("a model that learned little without collapsing", () => {
    const vit = prone("vit_b_16", vitEvidence, "adamw");

    it("does not say it predicted one class, in either language", () => {
      expect(note(en, vit)).toBe(
        "vit_b_16 learned little with Adam at 1e-3 (accuracy 0.41 on 4 classes). With adamw at 0.0001, accuracy was 0.85 under the same conditions.",
      );
      expect(note(pt, vit)).toBe(
        "vit_b_16 aprendeu pouco com Adam a 1e-3 (acurácia 0.41 em 4 classes). Com adamw a 0.0001, a acurácia foi 0.85 nas mesmas condições.",
      );
      expect(note(en, vit)).not.toMatch(/single class/);
      expect(note(pt, vit)).not.toMatch(/uma classe só/);
    });

    it("names the measured model for a sibling, without its recovery", () => {
      const vitL = prone("vit_l_16", vitEvidence, "adamw");
      expect(note(en, vitL)).toBe(
        "vit_l_16: vit_b_16, from the same family, learned little with Adam at 1e-3 (accuracy 0.41 on 4 classes); this model was not measured. We suggest adamw at 0.0001.",
      );
      expect(note(pt, vitL)).toBe(
        "vit_l_16: o vit_b_16, da mesma família, aprendeu pouco com Adam a 1e-3 (acurácia 0.41 em 4 classes); este modelo não foi medido. Sugerimos adamw a 0.0001.",
      );
    });
  });

  describe("a family that was never measured", () => {
    const maxvit = prone("maxvit_t", null, "adamw");

    it("quotes no accuracy, in either language", () => {
      expect(note(en, maxvit)).toBe(
        "maxvit_t: Adam at 1e-3 was not measured for this family; we suggest adamw at 0.0001, the same as the measured attention families.",
      );
      expect(note(pt, maxvit)).toBe(
        "maxvit_t: Adam a 1e-3 não foi medido para esta família; sugerimos adamw a 0.0001, o mesmo das famílias de atenção medidas.",
      );
    });
  });

  describe("a response without usable collapse_evidence", () => {
    // The field is simply absent; the flag alone must not bring a number back.
    const old: ModelDefaults = prone("vgg16", null);
    delete old.collapse_evidence;

    it("falls back to a plain suggestion, which claims nothing", () => {
      // "Not measured for this family ... the same as the measured attention
      // families" would be false for VGG, which was measured.
      expect(note(en, old)).toBe("For vgg16, we suggest adam at 0.0001.");
      expect(note(pt, old)).toBe("Para vgg16, sugerimos adam a 0.0001.");
    });

    it("never prints an accuracy the response did not carry", () => {
      expect(note(en, old)).not.toMatch(/0\.25|0\.41|accuracy/);
      expect(note(pt, old)).not.toMatch(/0\.25|0\.41|acurácia/);
    });

    it("does the same for an outcome this build does not know", () => {
      const future = prone("vgg16", {
        measured_on: "vgg16",
        accuracy: 0.25,
        outcome: "diverged" as CollapseEvidence["outcome"],
      });
      expect(note(en, future)).toBe("For vgg16, we suggest adam at 0.0001.");
      expect(note(pt, future)).toBe("Para vgg16, sugerimos adam a 0.0001.");
    });
  });

  describe("outside classification the evidence is labelled as classification's", () => {
    // The 0.25 / 0.41 / 0.85 were measured on classification. A regression or
    // segmentation form must not present them as its own, nor claim that the
    // suggested setting reached an accuracy "under the same conditions".
    const vgg16 = prone("vgg16", vggEvidence);
    const vgg19 = prone("vgg19", vggEvidence);
    const vit = prone("vit_b_16", vitEvidence, "adamw");
    const swin = prone("swin_t", swinEvidence, "adamw");

    it("says in English that it is classification, and that this task was not measured", () => {
      expect(note(en, vgg16, "regression")).toBe(
        "In classification, vgg16 predicted a single class with Adam at 1e-3 (accuracy 0.25 on 4 classes); this task was not measured. We suggest adam at 0.0001.",
      );
      expect(note(en, vit, "regression")).toBe(
        "In classification, vit_b_16 learned little with Adam at 1e-3 (accuracy 0.41 on 4 classes); this task was not measured. We suggest adamw at 0.0001.",
      );
    });

    it("says it in Portuguese too", () => {
      expect(note(pt, vgg16, "regression")).toBe(
        "Em classificação, o vgg16 previu uma classe só com Adam a 1e-3 (acurácia 0.25 em 4 classes); esta tarefa não foi medida. Sugerimos adam a 0.0001.",
      );
      expect(note(pt, vit, "regression")).toBe(
        "Em classificação, o vit_b_16 aprendeu pouco com Adam a 1e-3 (acurácia 0.41 em 4 classes); esta tarefa não foi medida. Sugerimos adamw a 0.0001.",
      );
    });

    it("names the sibling and the task as both unmeasured", () => {
      expect(note(en, vgg19, "regression")).toBe(
        "In classification, vgg16, from the same family as vgg19, predicted a single class with Adam at 1e-3 (accuracy 0.25 on 4 classes); neither vgg19 nor this task was measured. We suggest adam at 0.0001.",
      );
      expect(note(pt, vgg19, "regression")).toBe(
        "Em classificação, o vgg16, da mesma família do vgg19, previu uma classe só com Adam a 1e-3 (acurácia 0.25 em 4 classes); nem o vgg19 nem esta tarefa foram medidos. Sugerimos adam a 0.0001.",
      );
    });

    it("never carries the classification recovery over", () => {
      for (const task of ["regression", "segmentation"] as const) {
        expect(note(en, swin, task)).not.toMatch(/0\.88|same conditions/);
        expect(note(pt, swin, task)).not.toMatch(/0\.88|mesmas condições/);
        expect(note(en, swin, task)).toMatch(/We suggest adamw at 0\.0001\.$/);
      }
    });

    it("treats segmentation the same as regression", () => {
      expect(note(en, vgg16, "segmentation")).toBe(note(en, vgg16, "regression"));
      expect(note(pt, vit, "segmentation")).toBe(note(pt, vit, "regression"));
    });

    it("leaves the claim-free wordings alone", () => {
      const maxvit = prone("maxvit_t", null, "adamw");
      expect(note(en, maxvit, "regression")).toBe(note(en, maxvit));
      expect(note(en, base, "regression")).toBe("For resnet50, we suggest adam at 0.001.");
    });

    it("does not raise the alarm for evidence from another task", () => {
      expect(isAlarming(vgg16, "classification")).toBe(true);
      expect(isAlarming(vgg16, "regression")).toBe(false);
      expect(isAlarming(vgg16, "segmentation")).toBe(false);
    });
  });

  describe("an architecture that was never flagged", () => {
    it("suggests the setting instead of calling it measured, in both languages", () => {
      // resnet18, mobilenet and the rest were never in the measured grid, and
      // the note is shown for any of them whose form differs from the suggestion.
      for (const architecture of ["resnet50", "resnet18", "mobilenet_v3_small"]) {
        const advice = { ...base, architecture };
        expect(note(en, advice)).toBe(`For ${architecture}, we suggest adam at 0.001.`);
        expect(note(pt, advice)).toBe(`Para ${architecture}, sugerimos adam a 0.001.`);
      }
    });
  });

  it("warns about upscaling when the images are smaller than the suggested size", () => {
    const small: ModelDefaults = { ...base, image_size: 224, dataset_median_side: 32 };
    expect(note(pt, small)).toBe(
      "As imagens têm cerca de 32px de lado; treinar acima disso amplia a imagem sem acrescentar detalhe.",
    );
    expect(note(en, small)).toBe(
      "The images are about 32px on a side; training above that upsizes the image without adding detail.",
    );
  });

  it("puts the collapse warning ahead of the upscaling one", () => {
    const vgg16 = { ...prone("vgg16", vggEvidence), image_size: 224, dataset_median_side: 32 };
    expect(note(en, vgg16)).toMatch(/^vgg16 predicted a single class with Adam at 1e-3/);
  });

  it("stays quiet about upscaling when the images are as large as the size", () => {
    expect(note(en, { ...base, image_size: 224, dataset_median_side: 224 })).toBe(
      "For resnet50, we suggest adam at 0.001.",
    );
    expect(note(en, { ...base, image_size: null, dataset_median_side: 32 })).toBe(
      "For resnet50, we suggest adam at 0.001.",
    );
  });
});

describe("isAlarming", () => {
  it("is true where a failure was measured", () => {
    expect(isAlarming(prone("vgg16", vggEvidence), "classification")).toBe(true);
    expect(isAlarming(prone("vit_l_16", vitEvidence), "classification")).toBe(true);
  });

  it("is false for a flagged family with no evidence: the suggestion stays, the alarm does not", () => {
    expect(isAlarming(prone("maxvit_t", null, "adamw"), "classification")).toBe(false);
  });

  it("is false for a response that predates collapse_evidence", () => {
    const old: ModelDefaults = prone("vgg16", null);
    delete old.collapse_evidence;
    expect(isAlarming(old, "classification")).toBe(false);
  });

  it("is false for an outcome this build does not know", () => {
    const future = prone("vgg16", {
      measured_on: "vgg16",
      accuracy: 0.25,
      outcome: "diverged" as CollapseEvidence["outcome"],
    });
    expect(isAlarming(future, "classification")).toBe(false);
  });

  it("is false when nothing was flagged", () => {
    expect(isAlarming(base, "classification")).toBe(false);
  });
});
