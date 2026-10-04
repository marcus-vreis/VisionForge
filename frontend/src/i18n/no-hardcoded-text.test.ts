/**
 * Every word on screen comes from src/i18n/.
 *
 * Parsed with the TypeScript compiler rather than grepped, so comments,
 * imports and CSS values never count, and JSX text is told apart from code.
 * NOT_YET_MIGRATED lists the files still being moved over; the test fails
 * when a listed file is already clean, so the list can only shrink.
 */
/// <reference types="node" />
import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";
import ts from "typescript";
import { describe, expect, it } from "vitest";

const SRC = join(__dirname, "..");

const NOT_YET_MIGRATED = new Set([
  "App.tsx",
  "api/client.ts",
  "components/AdvancedFields.tsx",
  "components/AnomalyPanel.tsx",
  "components/AugmentPreview.tsx",
  "components/BottomBar.tsx",
  "components/CompareRunsPanel.tsx",
  "components/CredentialField.tsx",
  "components/CustomTaskManageCard.tsx",
  "components/CustomTaskPanel.tsx",
  "components/CvCard.tsx",
  "components/DatasetDownloadCard.tsx",
  "components/DatasetPicker.tsx",
  "components/DatasetStats.tsx",
  "components/DatasetsOverlay.tsx",
  "components/DetectionDatasetStats.tsx",
  "components/DetectionPanel.tsx",
  "components/DeviceSelector.tsx",
  "components/ExperimentHeader.tsx",
  "components/ExperimentRunner.tsx",
  "components/GuidedTour.tsx",
  "components/HistoryOverlay.tsx",
  "components/Lightbox.tsx",
  "components/ModelAdvice.tsx",
  "components/ParamPanel.tsx",
  "components/PreprocessingPanel.tsx",
  "components/QueueOverlay.tsx",
  "components/RegressionPanel.tsx",
  "components/ReplicatesCard.tsx",
  "components/ResultsView.tsx",
  "components/RunDetailPanel.tsx",
  "components/SchemaForm.tsx",
  "components/SegmentationPanel.tsx",
  "components/SweepCard.tsx",
  "components/TaskDatasetStats.tsx",
  "components/TaskHero.tsx",
  "components/TrainingOverlay.tsx",
  "components/TransformsSection.tsx",
  "components/WelcomeOverlay.tsx",
  "components/controls/InfoDot.tsx",
  "components/controls/WorkersField.tsx",
  "hooks/useExperiment.ts",
  "lib/anomaly-models.ts",
  "lib/dataset-identity.ts",
  "lib/grid-axis.ts",
  "lib/param-help.ts",
  "lib/queue-format.ts",
  "lib/regression-models.ts",
  "lib/run-notify.ts",
  "lib/segmentation-models.ts",
  "lib/tour.ts",
  "lib/yaml-config.ts",
  "types/tasks.ts",
]);

/** Exact texts that read the same in every language. */
const ALLOWED = new Set([
  // The header writes the brand as two spans.
  "VisionForge", "Vision", "Forge", "local ai studio",
  "CPU", "GPU", "CUDA", "MPS", "ONNX", "YOLO",
  "AUROC", "mIoU", "F1", "R²", "RMSE", "MAE", "MSE", "Dice",
  "px", "×", "·", "auto",
]);

const VISIBLE_PROPS = new Set([
  "title", "placeholder", "aria-label", "alt", "label", "subtitle",
  "help", "caption", "hint", "description", "tooltip", "emptyText",
]);

const PORTUGUESE =
  /[ãõçáéíóúâêôàÃÕÇÁÉÍÓÚÂÊÔÀ]|\b(nenhum|nenhuma|carregando|treinar|salvar|escolha|pasta|arquivo|você|voltar|continuar|pular|abrir|fechar|limpar|baixar|rodar|parar|retomar|apagar|dispositivos?|imagens?|de|do|da|dos|das|para|com|sem|uma?)\b/i;

const HAS_LETTER = /[A-Za-zÀ-ÿ]/;

interface Finding {
  file: string;
  line: number;
  text: string;
}

function sourceFiles(dir: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name !== "i18n") sourceFiles(path, out);
    } else if (/\.tsx?$/.test(entry.name) && !/\.test\.tsx?$/.test(entry.name) && !entry.name.endsWith(".d.ts")) {
      out.push(path);
    }
  }
  return out;
}

function findings(path: string): Finding[] {
  const text = readFileSync(path, "utf8");
  const kind = path.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
  const sf = ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true, kind);
  const file = relative(SRC, path).replace(/\\/g, "/");
  const out: Finding[] = [];
  const add = (node: ts.Node, s: string) =>
    out.push({ file, line: sf.getLineAndCharacterOfPosition(node.getStart()).line + 1, text: s.trim().slice(0, 60) });

  const visit = (node: ts.Node): void => {
    if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) return;
    if (ts.isJsxText(node)) {
      const s = node.text.trim();
      if (HAS_LETTER.test(s) && !ALLOWED.has(s)) add(node, s);
    } else if (
      ts.isJsxAttribute(node) &&
      node.initializer &&
      ts.isStringLiteral(node.initializer) &&
      VISIBLE_PROPS.has(node.name.getText())
    ) {
      const s = node.initializer.text.trim();
      if (HAS_LETTER.test(s) && !ALLOWED.has(s)) add(node, s);
      return;
    } else if (
      ts.isStringLiteral(node) ||
      ts.isNoSubstitutionTemplateLiteral(node) ||
      ts.isTemplateHead(node) ||
      ts.isTemplateMiddle(node) ||
      ts.isTemplateTail(node)
    ) {
      const s = node.text;
      if (s.trim() && (PORTUGUESE.test(s) || s === "pt-BR")) add(node, s);
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  return out;
}

describe("no hard-coded interface text", () => {
  const byFile = new Map(sourceFiles(SRC).map((p) => [relative(SRC, p).replace(/\\/g, "/"), findings(p)]));

  it("migrated files take every word from the dictionaries", () => {
    const offending = [...byFile]
      .filter(([file]) => !NOT_YET_MIGRATED.has(file))
      .flatMap(([, list]) => list)
      .map((f) => `${f.file}:${f.line}  ${f.text}`);
    expect(offending, "move these into src/i18n/pt.ts and en.ts").toEqual([]);
  });

  it("the not-yet-migrated list only holds files that still need it", () => {
    const done = [...NOT_YET_MIGRATED].filter((file) => (byFile.get(file) ?? []).length === 0);
    expect(done, "remove these from NOT_YET_MIGRATED").toEqual([]);
  });
});
