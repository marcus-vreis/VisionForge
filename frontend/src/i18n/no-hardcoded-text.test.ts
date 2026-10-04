/**
 * Every word on screen comes from src/i18n/.
 *
 * Parsed with the TypeScript compiler rather than grepped, so comments,
 * imports and CSS values never count, and JSX text is told apart from code.
 * NOT_YET_MIGRATED lists the files still being moved over; the test fails
 * when a listed file is already clean, so the list can only shrink.
 */
import { readdirSync, readFileSync } from "node:fs";
import { join, relative } from "node:path";
import ts from "typescript";
import { describe, expect, it } from "vitest";

const SRC = join(__dirname, "..");

const NOT_YET_MIGRATED = new Set([
  "components/AnomalyPanel.tsx",
  "components/AugmentPreview.tsx",
  "components/CompareRunsPanel.tsx",
  "components/CredentialField.tsx",
  "components/CustomTaskManageCard.tsx",
  "components/CustomTaskPanel.tsx",
  "components/CvCard.tsx",
  "components/DatasetDownloadCard.tsx",
  "components/DatasetPicker.tsx",
  "components/DatasetStats.tsx",
  "components/DetectionDatasetStats.tsx",
  "components/DetectionPanel.tsx",
  "components/ExperimentHeader.tsx",
  "components/GuidedTour.tsx",
  "components/HistoryOverlay.tsx",
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
  "components/TrainingOverlay.tsx",
  "components/TransformsSection.tsx",
  "components/controls/Toggle.tsx",
  "hooks/useExperiment.ts",
  "lib/anomaly-models.ts",
  "lib/custom-tasks.ts",
  "lib/dataset-identity.ts",
  "lib/grid-axis.ts",
  "lib/queue-format.ts",
  "lib/regression-models.ts",
  "lib/replicates-form.ts",
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
  // PyTorch's multi-GPU wrapper, shown as the "Multi-GPU" device subtitle.
  "DataParallel",
  // The glyph on the help dot.
  "i",
  // The version prefix in the header: `· v${version}`.
  "· v",
]);

const VISIBLE_PROPS = new Set([
  "title", "placeholder", "aria-label", "alt", "label", "subtitle",
  "help", "caption", "hint", "description", "tooltip", "emptyText",
]);

const PORTUGUESE =
  /[ãõçáéíóúâêôàÃÕÇÁÉÍÓÚÂÊÔÀ]|\b(nenhum|nenhuma|carregando|treinar|salvar|escolha|pasta|arquivo|você|voltar|continuar|pular|abrir|fechar|limpar|baixar|rodar|parar|retomar|apagar|dispositivos?|imagens?|de|do|da|dos|das|para|com|sem|uma?|pel[oa]s?|menos)\b/i;

const HAS_LETTER = /[A-Za-zÀ-ÿ]/;
const PROSE = /[a-zà-ÿ]{2,}\s+[a-zà-ÿ]{2,}/;

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

/** The literals an expression in a JSX child or visible prop can put on screen. */
function shownLiterals(expr: ts.Expression | undefined): ts.Expression[] {
  if (!expr) return [];
  if (ts.isParenthesizedExpression(expr)) return shownLiterals(expr.expression);
  if (ts.isStringLiteral(expr) || ts.isNoSubstitutionTemplateLiteral(expr) || ts.isTemplateExpression(expr)) return [expr];
  if (ts.isConditionalExpression(expr)) return [...shownLiterals(expr.whenTrue), ...shownLiterals(expr.whenFalse)];
  if (ts.isBinaryExpression(expr)) {
    switch (expr.operatorToken.kind) {
      case ts.SyntaxKind.PlusToken:
        return [...shownLiterals(expr.left), ...shownLiterals(expr.right)];
      case ts.SyntaxKind.BarBarToken:
      case ts.SyntaxKind.QuestionQuestionToken:
      case ts.SyntaxKind.AmpersandAmpersandToken:
        return shownLiterals(expr.right);
    }
  }
  return [];
}

/** The fixed text of a literal; a template's `${}` holes read as spaces. */
function literalText(lit: ts.Expression): string {
  return ts.isTemplateExpression(lit)
    ? [lit.head.text, ...lit.templateSpans.map((s) => s.literal.text)].join(" ")
    : (lit as ts.StringLiteralLike).text;
}

function findings(path: string): Finding[] {
  const text = readFileSync(path, "utf8");
  const kind = path.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
  const sf = ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true, kind);
  const file = relative(SRC, path).replace(/\\/g, "/");
  const out: Finding[] = [];
  const add = (node: ts.Node, s: string) =>
    out.push({ file, line: sf.getLineAndCharacterOfPosition(node.getStart()).line + 1, text: s.trim().slice(0, 60) });
  // Literals already judged as on-screen text, so the Portuguese check below skips them.
  const seen = new Set<ts.Node>();

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
      ts.isJsxExpression(node) &&
      (ts.isJsxElement(node.parent) ||
        ts.isJsxFragment(node.parent) ||
        (ts.isJsxAttribute(node.parent) && VISIBLE_PROPS.has(node.parent.name.getText())))
    ) {
      // {"..."}, {`...${n}`}, {ok ? "..." : "..."}, {x || "..."}, title={"..."}: on screen like JSX text.
      for (const lit of shownLiterals(node.expression)) {
        seen.add(lit);
        if (ts.isTemplateExpression(lit)) [lit.head, ...lit.templateSpans.map((s) => s.literal)].forEach((p) => seen.add(p));
        const s = literalText(lit).trim();
        if (HAS_LETTER.test(s) && !ALLOWED.has(s)) add(lit, s);
      }
    } else if (
      ts.isPropertyAssignment(node) &&
      (ts.isIdentifier(node.name) || ts.isStringLiteral(node.name)) &&
      VISIBLE_PROPS.has(node.name.text)
    ) {
      // { label: "…", placeholder: "…" } in config objects: only prose (two lower-case words), so
      // model and dataset names such as "YOLO11-n" or "Faster R-CNN" stay out. What is not prose
      // still gets the Portuguese check below.
      for (const lit of shownLiterals(node.initializer)) {
        const s = literalText(lit).trim();
        if (PROSE.test(s) && !ALLOWED.has(s)) {
          seen.add(lit);
          if (ts.isTemplateExpression(lit)) [lit.head, ...lit.templateSpans.map((p) => p.literal)].forEach((p) => seen.add(p));
          add(lit, s);
        }
      }
    } else if (
      !seen.has(node) &&
      (ts.isStringLiteral(node) ||
        ts.isNoSubstitutionTemplateLiteral(node) ||
        ts.isTemplateHead(node) ||
        ts.isTemplateMiddle(node) ||
        ts.isTemplateTail(node))
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
