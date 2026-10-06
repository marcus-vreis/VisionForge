import { describe, expect, it } from "vitest";
import { en } from "./en";
import { pt } from "./pt";

type Tree = { [key: string]: unknown };

function leaves(tree: Tree, prefix = ""): Array<[string, unknown]> {
  return Object.entries(tree).flatMap(([key, value]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return typeof value === "object" && value !== null
      ? leaves(value as Tree, path)
      : [[path, value] as [string, unknown]];
  });
}

describe("dictionaries", () => {
  const ptLeaves = new Map(leaves(pt));
  const enLeaves = new Map(leaves(en));

  it("have the same keys", () => {
    expect([...enLeaves.keys()].sort()).toEqual([...ptLeaves.keys()].sort());
  });

  it("have no empty text", () => {
    for (const [path, value] of [...ptLeaves, ...enLeaves]) {
      if (typeof value === "string") expect(value.trim(), path).not.toBe("");
    }
  });

  it("agree on which entries take values", () => {
    for (const [path, value] of ptLeaves) {
      expect(typeof enLeaves.get(path), path).toBe(typeof value);
      if (typeof value === "function") {
        expect((enLeaves.get(path) as (...a: unknown[]) => string).length, path).toBe(
          (value as (...a: unknown[]) => string).length,
        );
      }
    }
  });

  it("word a count with the plural right in both languages", () => {
    expect(pt.detectionDatasetStats.applied(1)).toBe("1 classe aplicada");
    expect(pt.detectionDatasetStats.applied(3)).toBe("3 classes aplicadas");
    expect(en.detectionDatasetStats.applied(1)).toBe("1 class applied");
    expect(en.detectionDatasetStats.applied(3)).toBe("3 classes applied");

    expect(pt.paramPanel.fieldErrors(1)).toBe("1 campo com erro:");
    expect(pt.paramPanel.fieldErrors(3)).toBe("3 campos com erro:");
    expect(en.paramPanel.fieldErrors(1)).toBe("1 field with errors:");
    expect(en.paramPanel.fieldErrors(3)).toBe("3 fields with errors:");

    const warnings = (dict: typeof pt, n: number) =>
      dict.paramPanel.importWarnings(n, "x", 0).split(":")[0];
    expect(warnings(pt, 1)).toBe("YAML importado com 1 aviso estrutural");
    expect(warnings(pt, 2)).toBe("YAML importado com 2 avisos estruturais");
    expect(warnings(en, 1)).toBe("YAML imported with 1 structural warning");
    expect(warnings(en, 2)).toBe("YAML imported with 2 structural warnings");

    expect(pt.runDetail.gradcam.done(1, "layer4")).toBe("1 mapa gerado · camada layer4");
    expect(pt.runDetail.gradcam.done(3, "layer4")).toBe("3 mapas gerados · camada layer4");
    expect(en.runDetail.gradcam.done(1, "layer4")).toBe("1 map generated · target layer: layer4");
    expect(en.runDetail.gradcam.done(3, "layer4")).toBe("3 maps generated · target layer: layer4");
  });

  it("agree with the number in the counts that can read 1 on screen", () => {
    expect(pt.paramPanel.grid.axisTag(1)).toBe("grade · 1 valor");
    expect(pt.paramPanel.grid.axisTag(3)).toBe("grade · 3 valores");
    expect(en.paramPanel.grid.axisTag(1)).toBe("grid · 1 value");
    expect(en.paramPanel.grid.axisTag(3)).toBe("grid · 3 values");

    expect(pt.paramPanel.hiddenParams(1)).toBe("1 parâmetro oculto — ligue para ajustar");
    expect(pt.paramPanel.hiddenParams(3)).toBe("3 parâmetros ocultos — ligue para ajustar");
    expect(en.paramPanel.hiddenParams(1)).toBe("1 hidden parameter — turn on to adjust it");
    expect(en.paramPanel.hiddenParams(3)).toBe("3 hidden parameters — turn on to adjust them");

    expect(pt.runDetail.dataset.files(1, "2 MB")).toBe("1 arquivo · 2 MB");
    expect(pt.runDetail.dataset.files(3, "2 MB")).toBe("3 arquivos · 2 MB");
    expect(pt.runDetail.dataset.files(null, "2 MB")).toBe("— arquivos · 2 MB");
    expect(en.runDetail.dataset.files(1, "2 MB")).toBe("1 file · 2 MB");
    expect(en.runDetail.dataset.files(3, "2 MB")).toBe("3 files · 2 MB");

    expect(pt.runDetail.batch.done(1, "a.csv")).toBe("1 imagem processada · CSV em a.csv");
    expect(pt.runDetail.batch.done(3, "a.csv")).toBe("3 imagens processadas · CSV em a.csv");
    expect(en.runDetail.batch.done(1, "a.csv")).toBe("1 image processed · CSV at a.csv");
    expect(en.runDetail.batch.done(3, "a.csv")).toBe("3 images processed · CSV at a.csv");

    // The noun agrees with the number of missing images, not with the number checked.
    expect(pt.taskDatasetStats.missingImages(1, 5)).toBe("⚠ 1/5 imagem não encontrada");
    expect(pt.taskDatasetStats.missingImages(2, 5)).toBe("⚠ 2/5 imagens não encontradas");
    expect(pt.taskDatasetStats.missingImages(1, 1)).toBe("⚠ 1/1 imagem não encontrada");
    expect(en.taskDatasetStats.missingImages(1, 5)).toBe("⚠ 1/5 image not found");
    expect(en.taskDatasetStats.missingImages(2, 5)).toBe("⚠ 2/5 images not found");
  });

  it("agree with the number in the count of runs that could not be deleted", () => {
    expect(pt.history.confirm.failed(1, 1)).toBe("1 de 1 não pôde ser excluído:");
    expect(pt.history.confirm.failed(2, 3)).toBe("2 de 3 não puderam ser excluídos:");
    expect(en.history.confirm.failed(1, 1)).toBe("1 of 1 run couldn't be deleted:");
    expect(en.history.confirm.failed(1, 3)).toBe("1 of 3 runs couldn't be deleted:");
    expect(en.history.confirm.failed(2, 3)).toBe("2 of 3 runs couldn't be deleted:");
  });

  it("name an unnamed run so the notification does not say the same word twice", () => {
    expect(pt.runNotify.completedTitle(pt.app.unnamedRun)).toBe("Treino concluído — sem nome");
    expect(en.runNotify.completedTitle(en.app.unnamedRun)).toBe("Training finished — unnamed run");
  });
});

describe("English terminology", () => {
  const texts = [...leaves(en)]
    .filter((entry): entry is [string, string] => typeof entry[1] === "string");

  it("spells in the American way", () => {
    for (const [path, text] of texts) expect(text, path).not.toMatch(/cancelled|cancelling/i);
  });

  it("writes an ellipsis as one character", () => {
    for (const [path, text] of texts) expect(text, path).not.toMatch(/[A-Za-z]\.\.\.($|\s)/);
  });

  it("gives a field one name wherever it is asked for", () => {
    expect(en.cvCard.foldSeed).toBe(en.paramPanel.fieldLabels.fold_seed);
    expect(en.cvCard.folds).toBe(en.paramPanel.fieldLabels.n_folds);
    expect(en.taskPanel.dataset.trainSplit).toBe(en.datasetPicker.trainSubdir);
    expect(en.taskPanel.dataset.valSplit).toBe(en.datasetPicker.valSubdir);
    expect(en.taskPanel.dataset.testSplit).toBe(en.datasetPicker.testSubdir);
    expect(en.taskPanel.dataset.trainSplit).toBe(en.paramPanel.fieldLabels.train_dir);
    expect(en.paramPanel.fieldLabels.weights_path).toBe(en.paramPanel.weights.label);
    expect(en.compareRuns.metrics.total_epochs).toBe(en.resultsView.metricLabels.total_epochs);
    expect(en.trainingOverlay.blocks.crossValidation).toBe(
      en.paramPanel.blocks.crossValidation.replace("(CV)", "CV"),
    );
  });

  it("words a failed operation as 'Failed to …', not 'Could not …'", () => {
    for (const [path, text] of texts) expect(text, path).not.toMatch(/^Could not\b/);
  });
});
