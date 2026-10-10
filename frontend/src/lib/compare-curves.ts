/**
 * Which per-epoch curves the History comparison can draw, per task.
 *
 * run.json `history` is one record per epoch, and each trainer writes its own
 * names into it: a classification epoch has `val_loss` and `val_accuracy`, a
 * detection epoch `map50_95` and `val_box_loss`, a regression epoch `val_r2`, a
 * segmentation epoch `val_miou`, an anomaly epoch `val_auroc` (which trainer
 * writes which is listed beside each task below). Charting two fixed names drew
 * nothing for every task but classification, so the series come from the task
 * of the runs being compared and from what their histories actually measured.
 *
 * Pure, and free of text: a series carries the dictionary key of its label
 * (`compareRuns.curves`), or none when the name is a researcher's own.
 */
import type { Dict } from "../i18n/pt";
import {
  isCustomTaskKey,
  metricDirection,
  numericMetric,
  type MetricDirection,
} from "./compare-metrics";

/** Keys of `compareRuns.curves`: the dictionary's name for a history key. */
export type CurveLabelKey = keyof Dict["compareRuns"]["curves"];

export interface CurveSeries {
  /** The name in a run.json `history` record. */
  key: string;
  /** The dictionary's label; null where the series' own name is the label. */
  label: CurveLabelKey | null;
  /** Which end is better. Null for a decision point, which is neither. */
  direction: MetricDirection | null;
  /** Charted before the researcher picks anything. */
  initial: boolean;
}

interface TaskCurves {
  /** What is charted at first, in order. Each entry lists alternatives for the
   *  same thing; the first one the runs measured is the one used. */
  initial: CurveLabelKey[][];
  /** Also offered when measured. */
  more: CurveLabelKey[];
}

/**
 * The series of each built-in task, and where its history writes them:
 * classification `core/trainer.py` (`_write_run_json`, the `history` list),
 * detection `core/detection_trainer.py`, regression `core/regression_trainer.py`,
 * segmentation `core/segmentation_trainer.py`, anomaly `core/anomaly_trainer.py`
 * (each in its own `_write_run_json`).
 *
 * Detection's `box_loss` is the validation box loss under its first name (both
 * backends write the same number to `val_box_loss`), so it stands in for
 * `val_box_loss` on a history that predates it and is never listed beside it.
 * The torchvision backend measures no mAP@50-95, so mAP@50 stands in for it.
 */
const CURVES: Record<string, TaskCurves> = {
  classification: {
    initial: [["val_loss"], ["val_accuracy"]],
    more: ["train_loss", "train_accuracy"],
  },
  detection: {
    initial: [["map50_95", "map50"], ["val_box_loss", "box_loss"]],
    more: [
      "map50",
      "precision",
      "recall",
      "train_box_loss",
      "train_cls_loss",
      "train_dfl_loss",
      "val_cls_loss",
      "val_dfl_loss",
    ],
  },
  regression: {
    initial: [["val_loss"], ["val_r2"]],
    more: ["val_rmse", "val_mae", "val_mse", "train_loss"],
  },
  segmentation: {
    initial: [["val_loss"], ["val_miou"]],
    more: ["val_dice", "val_pixel_acc", "train_loss"],
  },
  // PatchCore fits in one epoch and writes a train loss of zero: the score is
  // the curve that says something, so it is the one charted first.
  anomaly: {
    initial: [["val_auroc"]],
    more: ["val_image_f1", "train_loss", "val_threshold"],
  },
};

/** A decision point per epoch is neither better high nor low; any other series
 *  improves the way the server said (`RunDetail.metric_directions`), else by name. */
function directionOf(
  key: string,
  served: Readonly<Record<string, string>> | undefined,
): MetricDirection | null {
  return key.endsWith("threshold") ? null : metricDirection(key, served);
}

type History = ReadonlyArray<Record<string, unknown>>;

/**
 * The curves the runs can draw, in the order they are offered: the ones charted
 * first, then the rest of the task's. Only a series at least one run measured at
 * least once: a metric a backend leaves null (torchvision has no precision) is
 * not offered. Empty when no run kept a history (a replicate group keeps none).
 *
 * A task that is not built in (a researcher's own, ADR-058) offers every
 * numeric key its histories carry, under the names it reported, and charts its
 * train loss and its first validation metric. A task this module does not know
 * is read as classification, as the History card does.
 */
export function curveSeries(
  task: string,
  histories: ReadonlyArray<History>,
  served?: Readonly<Record<string, string>>,
): CurveSeries[] {
  const measured = (key: string) =>
    histories.some((history) => history.some((record) => numericMetric(record[key]) !== null));

  if (isCustomTaskKey(task)) {
    const keys = [
      ...new Set(histories.flatMap((history) => history.flatMap((record) => Object.keys(record)))),
    ].filter((key) => key !== "epoch" && measured(key));
    const first = keys.find((key) => key.startsWith("val_"));
    return keys.map((key) => ({
      key,
      label: null,
      direction: directionOf(key, served),
      initial: key === "train_loss" || key === first,
    }));
  }

  const curves = CURVES[task] ?? CURVES.classification;
  const initial = curves.initial
    .map((alternatives) => alternatives.find(measured))
    .filter((key): key is CurveLabelKey => key !== undefined);
  const keys = [...new Set<CurveLabelKey>([...initial, ...curves.more.filter(measured)])];
  return keys.map((key) => ({
    key,
    label: key,
    direction: directionOf(key, served),
    initial: initial.includes(key),
  }));
}

/**
 * The series to chart: the ones picked, or the initial ones while nothing is
 * picked or what was picked is not among these runs' series any more (a
 * different comparison). Never empty when there is something to draw.
 */
export function selectedCurves(
  series: ReadonlyArray<CurveSeries>,
  picked: ReadonlyArray<string> | null,
): CurveSeries[] {
  const chosen = picked ? series.filter((s) => picked.includes(s.key)) : [];
  return chosen.length > 0 ? chosen : series.filter((s) => s.initial);
}

/** The picks after clicking `key`: added when absent, removed when present, and
 *  unchanged when removing it would leave nothing to chart. */
export function toggleCurve(current: ReadonlyArray<string>, key: string): string[] {
  if (!current.includes(key)) return [...current, key];
  return current.length > 1 ? current.filter((k) => k !== key) : [...current];
}
