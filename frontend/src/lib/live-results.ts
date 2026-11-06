/** What the "Resultados" tab of the training sheet shows, read off the event
 * stream (docs/dev/design-resultados-ao-vivo-e-colab.md, phase A).
 *
 * Nothing here costs the training anything: the curves are the `epoch_end`
 * events the sheet already receives, and the best epoch is found with the same
 * rule the trainer uses to pick the checkpoint it saves, so the card names the
 * epoch whose weights `best_model.pth` holds.
 *
 * Pure on purpose: no DOM and no fetch, so it is covered by plain vitest.
 */

import type { LiveConfusion, TrainingEvent } from "../types/run";
import type { MetricDirection } from "./compare-metrics";
import type { MetricKind, StorySpec } from "./story-family";

export type EpochEnd = Extract<TrainingEvent, { event: "epoch_end" }>;
export type InsightEvent = Extract<TrainingEvent, { event: "insight" }>;

/** One training of the job: the whole run, or one trial, fold, model or
 *  replicate of a multi-unit job. */
export interface LiveUnit {
  /** 0-based, from `trial_start`; null for a run of a single training. */
  index: number | null;
  /** What the unit changed (`trial_start.overrides`); null for a single run. */
  overrides: Record<string, unknown> | null;
  epochs: EpochEnd[];
  /** The unit reported its `trial_end`. */
  closed: boolean;
  /** Its latest Grad-CAM (ADR-162), when the run asked for them. */
  insight: InsightEvent | null;
}

/** The units of the job in the order they ran. A run without `trial_start`
 *  is one unit. */
export function liveUnits(events: readonly TrainingEvent[]): LiveUnit[] {
  const units: LiveUnit[] = [];
  let current: LiveUnit | undefined;
  for (const e of events) {
    if (e.event === "trial_start") {
      current = {
        index: e.trial_index,
        overrides: e.overrides,
        epochs: [],
        closed: false,
        insight: null,
      };
      units.push(current);
    } else if (e.event === "epoch_end") {
      if (current === undefined) {
        current = { index: null, overrides: null, epochs: [], closed: false, insight: null };
        units.push(current);
      }
      current.epochs.push(e);
    } else if (e.event === "trial_end" && current !== undefined) {
      current.closed = true;
    } else if (e.event === "insight" && current !== undefined) {
      current.insight = e;
    }
  }
  return units;
}

/** Which number of an epoch picks the saved checkpoint, and which end wins. */
export interface BestRule {
  key: string;
  direction: MetricDirection;
}

function finite(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * The rule each trainer applies when it saves `best_model.pth`:
 *
 * - classification `core/trainer.py` and regression `core/regression_trainer.py`:
 *   the lowest `val_loss` (`MetricPolicy("val_loss")`);
 * - segmentation `core/segmentation_trainer.py`: the highest `val_miou`;
 * - detection `core/detection_trainer.py`: Ultralytics keeps the highest
 *   mAP@50-95; torchvision measures no mAP@50-95 (it reports null) and keeps the
 *   highest mAP@50;
 * - anomaly `core/anomaly_trainer.py`: the autoencoder keeps the lowest training
 *   loss. PatchCore fits in one pass and has no epoch to choose.
 *
 * A researcher's own task (`tasks/engine.py`) declares the direction of its
 * primary metric in its task info, which the stream does not carry, so there is
 * no rule to state for it: null rather than a guess.
 */
export function bestRule(spec: StorySpec, epochs: readonly EpochEnd[]): BestRule | null {
  switch (spec.task) {
    case "classification":
    case "regression":
      return { key: "val_loss", direction: "lower" };
    case "segmentation":
      return { key: "val_miou", direction: "higher" };
    case "detection":
      return epochs.some((e) => finite(e.map50_95) !== null)
        ? { key: "map50_95", direction: "higher" }
        : { key: "map50", direction: "higher" };
    case "anomaly":
      return spec.family === "autoencoder" ? { key: "train_loss", direction: "lower" } : null;
    default:
      return null;
  }
}

export interface BestEpoch {
  epoch: number;
  key: string;
  value: number;
  direction: MetricDirection;
}

/** The epoch the rule keeps. The trainers replace the checkpoint only on a strict
 *  improvement, so a tie keeps the earlier epoch. */
export function bestEpoch(epochs: readonly EpochEnd[], rule: BestRule | null): BestEpoch | null {
  if (rule === null) return null;
  let best: BestEpoch | null = null;
  for (const e of epochs) {
    const value = finite(e[rule.key as keyof EpochEnd]);
    if (value === null) continue;
    const better =
      best === null || (rule.direction === "higher" ? value > best.value : value < best.value);
    if (better) best = { epoch: e.epoch, key: rule.key, value, direction: rule.direction };
  }
  return best;
}

export interface CurvePoint {
  epoch: number;
  value: number;
}

/** The finite values of one number across the epochs. */
export function curveOf(epochs: readonly EpochEnd[], key: string): CurvePoint[] {
  const points: CurvePoint[] = [];
  for (const e of epochs) {
    const value = finite(e[key as keyof EpochEnd]);
    if (value !== null) points.push({ epoch: e.epoch, value });
  }
  return points;
}

/** The `epoch_end` field that carries the task's headline metric. */
export function metricKey(kind: MetricKind): string {
  switch (kind) {
    case "map50":
      return "map50";
    case "r2":
      return "val_r2";
    case "miou":
      return "val_miou";
    case "auroc":
      return "val_auroc";
    default:
      // `primary` is a researcher's own task: its value arrives as the compat
      // field `val_accuracy` (ADR-058).
      return "val_accuracy";
  }
}

/** What a unit changed, as short `name=value` pairs. */
export function overridesSummary(overrides: Record<string, unknown> | null): string {
  if (overrides === null) return "";
  return Object.entries(overrides)
    .map(([name, value]) =>
      `${name}=${typeof value === "object" && value !== null ? JSON.stringify(value) : String(value)}`,
    )
    .join(", ");
}

/** Four decimals hide a loss that trained down to 2e-4 as "0.0000", so the
 *  tiny ones are written in scientific notation. */
export function formatValue(value: number): string {
  const size = Math.abs(value);
  if (size === 0) return "0";
  if (size < 1e-3) return value.toExponential(2);
  return size >= 100 ? value.toFixed(1) : value.toFixed(4);
}

/** The confusion of the latest epoch that sent one, and that epoch. */
export function latestConfusion(
  epochs: readonly EpochEnd[],
): { epoch: number; confusion: LiveConfusion } | null {
  for (let i = epochs.length - 1; i >= 0; i -= 1) {
    const confusion = epochs[i].confusion;
    if (confusion && confusion.classes.length > 0) return { epoch: epochs[i].epoch, confusion };
  }
  return null;
}
