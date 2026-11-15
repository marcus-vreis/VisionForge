/** "E agora?": what to do after reading a run's numbers.
 *
 * At most two suggestions, each read from facts the run detail already
 * carries; none is a diagnosis. When no rule holds, nothing is suggested and
 * the block is not drawn.
 */

import type { Dict } from "../i18n/pt";
import { RERUN_SEEDS } from "./history-rerun";

/** The suggestions, in priority order. */
export type NextStepId = "unfinished" | "stillImproving" | "untested" | "singleRun";

/** The panel action a suggestion points to; the panel drops one it cannot offer. */
export type NextStepAction = "resume" | "reuse" | "test" | "replicate";

export interface NextStep {
  id: NextStepId;
  action: NextStepAction | null;
}

/** What the rules read from `GET /api/runs/{id}`. */
export interface NextStepsInput {
  status: string;
  stopped?: boolean;
  resumable?: boolean;
  metrics: Readonly<Record<string, unknown>>;
  tests: readonly unknown[];
  artifacts: { model?: string };
  group?: unknown;
  group_id?: string | null;
  task?: string;
  config: Readonly<Record<string, unknown>>;
}

export const MAX_NEXT_STEPS = 2;

// The classification blocks that train one model with one seed; a K-fold, a
// sweep or a comparison is several trainings (run_history.py reads the same key).
const SINGLE_TRAINING_BLOCKS = new Set([
  "classification",
  "transfer_learning",
  "regression",
  "segmentation",
  "anomaly",
  "detection",
]);

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isUnfinished(d: NextStepsInput): boolean {
  return d.status === "failed" || d.stopped === true || d.resumable === true;
}

/** The best checkpoint is the last epoch trained, after more than one. An early
 *  stop leaves the best `patience` epochs behind the last, so it never matches. */
function stillImproving(d: NextStepsInput): boolean {
  const best = d.metrics["best_epoch"];
  const total = d.metrics["total_epochs"];
  return !d.group && finiteNumber(best) && finiteNumber(total) && total > 1 && best === total;
}

/** A held-out number exists: the training's own test split, or a test run here. */
function hasTestResult(d: NextStepsInput): boolean {
  if (d.tests.length > 0) return true;
  return Object.entries(d.metrics).some(([key, value]) => key.startsWith("test_") && finiteNumber(value));
}

/** The test action needs a checkpoint the server can load: a group has none and
 *  a researcher's task owns its own loop (the API answers 400 to both). */
function canTest(d: NextStepsInput): boolean {
  return !d.group && Boolean(d.artifacts.model) && !(d.task ?? "").startsWith("custom:");
}

function isSingleTraining(d: NextStepsInput): boolean {
  if (d.group || d.group_id) return false;
  if (Array.isArray(d.metrics["fold_results"])) return false;
  const block = d.config["block"];
  return block === undefined || block === null || SINGLE_TRAINING_BLOCKS.has(String(block));
}

/** The first `MAX_NEXT_STEPS` suggestions that hold for this run; none while it trains. */
export function nextSteps(d: NextStepsInput): NextStep[] {
  if (d.status === "running") return [];
  const steps: NextStep[] = [];
  if (isUnfinished(d)) {
    steps.push({ id: "unfinished", action: d.resumable ? "resume" : null });
  } else if (d.status === "completed") {
    if (stillImproving(d)) steps.push({ id: "stillImproving", action: "reuse" });
    if (canTest(d) && !hasTestResult(d)) steps.push({ id: "untested", action: "test" });
    if (isSingleTraining(d)) steps.push({ id: "singleRun", action: "replicate" });
  }
  return steps.slice(0, MAX_NEXT_STEPS);
}

/** The panel's handlers, by action; a missing one means the panel cannot offer it here. */
export type NextStepHandlers = Partial<Record<NextStepAction, () => void>>;

/** One line of the block, as drawn. */
export interface NextStepItem {
  id: NextStepId;
  text: string;
  action: { kind: NextStepAction; label: string; onClick: () => void; disabled: boolean } | null;
}

// The same words the panel's own buttons for these actions use.
function actionLabel(t: Dict, action: NextStepAction): string {
  switch (action) {
    case "resume":
      return t.nextSteps.resume;
    case "reuse":
      return t.runDetail.rerun.load;
    case "test":
      return t.runDetail.tests.open;
    case "replicate":
      return t.runDetail.rerun.replicates(RERUN_SEEDS);
  }
}

/** The sentence and the button of each suggestion; a suggestion whose action the
 *  panel has no handler for keeps its sentence and loses the button. */
export function nextStepItems(
  t: Dict,
  steps: readonly NextStep[],
  handlers: NextStepHandlers,
  disabled: Partial<Record<NextStepAction, boolean>> = {},
): NextStepItem[] {
  return steps.map((step) => {
    const onClick = step.action ? handlers[step.action] : undefined;
    const action =
      step.action && onClick
        ? {
            kind: step.action,
            label: actionLabel(t, step.action),
            onClick,
            disabled: disabled[step.action] === true,
          }
        : null;
    const text =
      step.id === "unfinished" && action?.kind === "resume"
        ? t.nextSteps.unfinishedResume
        : t.nextSteps[step.id];
    return { id: step.id, text, action };
  });
}
