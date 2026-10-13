import type { Dict } from "../i18n/pt";

/**
 * The phase labels the PatchCore trainer sends (core/anomaly_trainer.py), which
 * are Portuguese, and the dictionary entry that words each one. The backend is
 * not translated, so the label is the key; anything else is shown as received.
 */
const KNOWN_PHASES = new Map<string, PhaseKey>([
  ["extraindo features", "extractingFeatures"],
  ["montando o banco", "buildingBank"],
  ["pontuando", "scoring"],
]);

/** The dictionary key of a phase the interface knows, `null` for any other label. */
export type PhaseKey = keyof Dict["trainingOverlay"]["phases"];

/** Which of the three known phases a label is, or `null`. */
export function phaseKey(label: string): PhaseKey | null {
  return KNOWN_PHASES.get(label) ?? null;
}

/** A phase label in the language of `t`; an unknown label is returned as is. */
export function phaseName(t: Dict, label: string): string {
  const key = KNOWN_PHASES.get(label);
  return key ? t.trainingOverlay.phases[key] : label;
}
