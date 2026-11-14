import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import type { ExplorerForm, ExplorerSummary } from "../lib/dataset-explorer";
import { DatasetExplorer, DatasetSummaryCard } from "./DatasetExplorer";
import { DatasetLab } from "./DatasetLab";

function render(lang: "pt" | "en", node: ReactNode) {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={{ lang, t: lang === "pt" ? pt : en, locale: lang, setLang: () => {} }}>
      {node}
    </LanguageContext.Provider>,
  );
}

const longPath = "C:/datasets/projetos/laboratorio-de-visao/2026/colheita/cafe-folhas-doentes";

const form = (task: ExplorerForm["task"] = "classification"): ExplorerForm => ({
  task,
  base_dir: longPath,
  train_dir: "train",
  val_dir: "val",
  test_dir: "test",
});

const split = (images: number, per_class: Record<string, number>, present = true) => ({
  present,
  images,
  per_class,
  boxes: {},
  unlabeled: 0,
});

function summary(over: Partial<ExplorerSummary> = {}): ExplorerSummary {
  return {
    base_dir: longPath,
    task: "classification",
    class_names: ["ferrugem", "saudavel"],
    total_images: 30,
    splits: {
      train: split(20, { ferrugem: 12, saudavel: 8 }),
      val: split(6, { ferrugem: 3, saudavel: 3 }),
      test: split(4, { ferrugem: 2, saudavel: 2 }),
    },
    ...over,
  };
}

const noop = () => {};

function lab(lang: "pt" | "en", props: Partial<Parameters<typeof DatasetLab>[0]> = {}) {
  return render(
    lang,
    <DatasetLab
      form={form()}
      summary={summary()}
      onClose={noop}
      onOpenResult={noop}
      onOpenDetectionResult={noop}
      {...props}
    />,
  );
}

/** The markup of the visible tab panel (the hidden ones are left out). */
function tabs(html: string): string[] {
  return [...html.matchAll(/role="tab"[^>]*>(.*?)<\/button>/g)].map((m) => m[1].replace(/<[^>]+>/g, ""));
}

describe("dataset lab dialog", () => {
  it.each(["pt", "en"] as const)("is a labelled modal dialog with the dataset in its header (%s)", (lang) => {
    const t = lang === "pt" ? pt : en;
    const html = lab(lang);
    expect(html).toContain('role="dialog"');
    expect(html).toContain('aria-modal="true"');
    expect(html).toContain(`aria-label="${t.datasetLab.dialogLabel("cafe-folhas-doentes")}"`);
    expect(html).toContain(t.datasetLab.back);
    expect(html).toContain(t.datasetLab.typeClassification);
    // The path is cut in the middle on screen and whole in the tooltip.
    expect(html).toContain(`title="${longPath}"`);
    expect(html).toContain("…");
    expect(html).toContain(t.datasetLab.images(30));
    expect(html).toContain(t.datasetLab.classes(2));
  });

  it("lists every section in the rail with the overview selected first", () => {
    const html = lab("pt");
    expect(tabs(html)).toEqual([
      pt.datasetLab.sections.overview,
      pt.datasetLab.sections.resplit,
      pt.datasetLab.sections.incorporate,
      pt.datasetLab.sections.merge,
      pt.datasetLab.sections.coco,
    ]);
    expect(html).toMatch(/aria-selected="true"[^>]*id="vf-lab-tab-overview"|id="vf-lab-tab-overview"[^>]*aria-selected="true"/);
    expect(html.match(/<svg/g)?.length ?? 0).toBeGreaterThanOrEqual(5);
  });

  it("shows only one tool: the others are not mounted until visited", () => {
    const html = lab("pt");
    expect(html).toContain('id="vf-lab-panel-overview"');
    expect(html).not.toContain('id="vf-lab-panel-resplit"');
    expect(html).not.toContain(pt.datasetLab.leadResplit);
  });

  it("lays the overview out as total cards and per-class bars with the count at the end", () => {
    const html = lab("pt");
    expect(html).toContain(pt.datasetLab.cardTotal);
    expect(html).toContain(pt.datasetLab.cardClasses);
    expect(html).toContain('class="vf-lab-stat__value">30<');
    expect(html).toContain(pt.datasetLab.perClassTitle);
    // ferrugem: 12 + 3 + 2, the largest class first.
    expect(html).toMatch(/ferrugem<\/span>.*?class="vf-lab-bar__value">17</);
    expect(html).toMatch(/saudavel<\/span>.*?class="vf-lab-bar__value">13</);
    expect(html).toContain(pt.datasetLab.segmentTitle(pt.datasetExplorer.splitNames.train, 12));
    // The exact class x split table stays, behind a closed disclosure.
    expect(html).toContain(pt.datasetLab.tableToggle);
    expect(html).not.toMatch(/<details[^>]*open/);
    expect(html).toContain(pt.datasetExplorer.scanTitle);
  });

  it("marks a missing split on its card", () => {
    const s = summary();
    s.splits.test = split(0, {}, false);
    expect(lab("pt", { summary: s })).toContain(pt.datasetLab.cardMissing);
  });
});

describe("lab tools", () => {
  it("re-split: one sentence, the long text behind saiba mais, the group CSV and the analyze action", () => {
    const html = lab("pt", { initial: "resplit" });
    expect(html).toContain(pt.datasetLab.leadResplit);
    expect(pt.datasetLab.leadResplit).toContain("o teste fica como está");
    expect(html).toContain(`<summary>${pt.datasetLab.learnMore}</summary>`);
    expect(html).toContain(pt.datasetResplit.intro);
    expect(html).toContain(pt.datasetLab.groupLead);
    expect(html).toContain(pt.datasetLab.groupMore);
    expect(html).toContain(pt.datasetResplit.groupFile);
    expect(html).toContain(pt.datasetResplit.analyze);
    // No more "open / close" toggle around the tool.
    expect(html).not.toContain(`>${pt.datasetResplit.close}<`);
    expect(html).not.toMatch(/<details[^>]*open/);
  });

  it("incorporate: the source format stays visible and analyze waits for a folder", () => {
    const html = lab("en", { initial: "incorporate" });
    expect(html).toContain(en.datasetLab.leadIncorporate);
    expect(html).toContain(en.datasetIncorporate.sourceClassification);
    expect(html).toMatch(new RegExp(`disabled=""[^>]*>(<span[^>]*>.*?</span>)?<span>${en.datasetIncorporate.analyze}<`));
  });

  it("merge: the same type rule stays visible", () => {
    const html = lab("pt", { initial: "merge" });
    expect(html).toContain(pt.datasetLab.leadMerge);
    expect(html).toContain(pt.datasetMerge.sourceHelp);
    expect(html).toContain(pt.datasetMerge.analyze);
  });

  it("COCO on a detection dataset offers export and import, one at a time", () => {
    const html = lab("pt", { form: form("detection"), summary: summary({ task: "detection" }), initial: "coco" });
    expect(html).toContain(pt.datasetLab.cocoMode);
    expect(html).toContain(pt.datasetLab.leadCocoExport);
    expect(html).toContain(pt.datasetLab.leadCocoImport);
    expect(html).toMatch(/<div hidden=""><section class="vf-lab-tool"><header class="vf-lab-tool__head"><h2 class="vf-lab-tool__title">Importar COCO/);
    expect(html).toContain(pt.datasetLab.typeDetection);
  });

  it("COCO on a classification dataset only imports, and says why", () => {
    const html = lab("pt", { initial: "coco" });
    expect(html).toContain(pt.datasetLab.cocoExportOnlyDetection);
    expect(html).not.toContain(pt.datasetLab.leadCocoExport);
    expect(html).toContain(pt.datasetLab.leadCocoImport);
  });

  it("without a dataset the lab only converts COCO", () => {
    const html = lab("en", { form: null, summary: null, initial: "overview" });
    expect(tabs(html)).toEqual([en.datasetLab.sections.coco]);
    expect(html).toContain(en.datasetLab.noDatasetHint);
    expect(html).toContain(en.datasetLab.leadCocoImport);
  });
});

describe("datasets overlay, explore tab", () => {
  it("only chooses the dataset: type, folder and open, no tools", () => {
    const html = render("pt", <DatasetExplorer />);
    expect(html).toContain(pt.datasetExplorer.task);
    expect(html).toContain(pt.datasetExplorer.open);
    for (const tool of [
      pt.datasetResplit.open,
      pt.datasetIncorporate.open,
      pt.datasetMerge.open,
      pt.datasetCoco.importOpen,
      pt.datasetCoco.exportOpen,
      pt.datasetLab.openLab,
    ]) {
      expect(html).not.toContain(tool);
    }
    expect(html).not.toContain('role="dialog"');
  });

  it("sums the chosen dataset up and offers the lab", () => {
    const html = render("pt", <DatasetSummaryCard form={form()} summary={summary()} onOpenLab={noop} />);
    expect(html).toContain("cafe-folhas-doentes");
    expect(html).toContain(`${pt.datasetExplorer.splitNames.train}: 20`);
    expect(html).toContain(`${pt.datasetExplorer.splitNames.test}: 4`);
    expect(html).toContain(pt.datasetLab.classes(2));
    expect(html).toContain(pt.datasetLab.openLab);
    expect(html).toContain("vf-action-btn--primary");
  });

  it("warns about a missing validation folder and keeps an empty dataset out of the lab", () => {
    const s = summary();
    s.splits.val = split(0, {}, false);
    const warned = render("en", <DatasetSummaryCard form={form()} summary={s} onOpenLab={noop} />);
    expect(warned).toContain(en.datasetLab.noteMissing(en.datasetExplorer.splitNames.val));
    expect(warned).toContain(en.datasetLab.openLab);

    const empty = summary({ total_images: 0, splits: {
      train: split(0, {}, false), val: split(0, {}, false), test: split(0, {}, false),
    } });
    const html = render("en", <DatasetSummaryCard form={form()} summary={empty} onOpenLab={noop} />);
    expect(html).toContain(en.datasetExplorer.empty);
    expect(html).not.toContain(en.datasetLab.openLab);
  });
});
