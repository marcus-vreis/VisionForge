/** The training sheet's log lines, one pass over the events that are new.
 *
 * Writing a line per render, from the latest event, drops lines whenever several
 * events land together, which is what a run opened mid-way receives (ADR-159).
 */

import type { Dict } from "../i18n/pt";
import type { TrainingEvent } from "../types/run";
import { trialBanner } from "./run-control";
import { phaseName } from "./training-phase";

export type LogLine = string | ((d: Dict) => string);

type EpochEnd = Extract<TrainingEvent, { event: "epoch_end" }>;

export function epochLogLine(event: EpochEnd): string {
  const { epoch, total_epochs, train_loss, val_loss, val_accuracy } = event;
  const trialTag =
    event.trial_index !== undefined && event.total_trials
      ? `[t${event.trial_index + 1}/${event.total_trials}] `
      : "";
  // Detection runs carry mAP / precision / per-component losses; show those
  // instead of the classification loss/acc triple. `map50` is present (maybe
  // null) only on detection events, so it discriminates the task here.
  const isDetection = event.map50 !== undefined;
  const fmt = (v: number | null | undefined): string =>
    v === null || v === undefined ? "—" : v.toFixed(4);
  // Researcher-defined tasks (ADR-058) stream their own declared metrics as
  // val_<name> and have no val_loss — list what the event actually carries
  // instead of assuming the classification triple (which crashed the
  // overlay when a custom task's first epoch arrived).
  const customMetrics = Object.entries(event)
    .filter(
      ([k, v]) =>
        k.startsWith("val_") &&
        k !== "val_accuracy" &&
        k !== "val_loss" &&
        !k.endsWith("_loss") &&
        typeof v === "number",
    )
    .map(([k, v]) => `${k.slice(4)}=${fmt(v as number)}`);

  const metricsPart = isDetection
    ? [
        `map50=${fmt(event.map50)}`,
        `map50-95=${fmt(event.map50_95)}`,
        `P=${fmt(event.precision)}`,
        `R=${fmt(event.recall)}`,
        `box=${fmt(event.val_box_loss ?? event.box_loss)}`,
        `cls=${fmt(event.val_cls_loss)}`,
        `dfl=${fmt(event.val_dfl_loss)}`,
      ].join(" · ")
    : customMetrics.length > 0
      ? [`loss=${fmt(train_loss)}`, ...customMetrics].join(" · ")
      : `loss=${fmt(train_loss)} · val_loss=${fmt(val_loss)} · val_acc=${fmt(val_accuracy)}`;
  return `> ${trialTag}epoch ${epoch}/${total_epochs} · ${metricsPart}`;
}

/** Lines for `events`, in order, and the phase the log stands in afterwards.
 *
 * One line per phase, not per update: building a memory bank reports ~100 times,
 * and a hundred equal lines would push everything useful out of the log.
 */
export function logLinesFor(
  events: readonly TrainingEvent[],
  currentPhase: string | null,
): { lines: LogLine[]; phase: string | null } {
  const lines: LogLine[] = [];
  let phase = currentPhase;
  for (const event of events) {
    if (event.event === "trial_start") {
      lines.push(trialBanner(event.trial_index, event.total_trials, event.overrides));
    } else if (event.event === "epoch_end") {
      lines.push(epochLogLine(event));
    } else if (event.event === "phase" && event.label !== phase) {
      const label = event.label;
      phase = label;
      // A function of the dictionary, so a known label follows the language.
      lines.push((d) => `> ${phaseName(d, label)}…`);
    }
  }
  return { lines, phase };
}
