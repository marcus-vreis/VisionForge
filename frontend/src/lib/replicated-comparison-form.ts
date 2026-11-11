/** Pure helpers for the ReplicatedComparisonCard (ADR-193): variants on one
 *  axis, the shared seeds, and the request the backend's
 *  /{task}/replicated-comparison route takes (ADR-061). Kept out of the
 *  component so the rules that disable the card are unit-testable. */

import type { Dict } from "../i18n/pt";
import { parseSeeds, seedsProblem, type ReplicatesSeedMode } from "./replicates-form";

/** The backend caps `n_replicates` at 20 (ReplicatedComparisonRequest); the GUI
 *  holds an explicit list to the same cap, since the cost is variants × seeds. */
export const MAX_COMPARISON_SEEDS = 20;
/** The form is one variant; one axis adds 1 to 3 values. */
export const MIN_COMPARISON_VARIANTS = 2;
export const MAX_COMPARISON_VARIANTS = 4;
/** The backend's default alpha; the card does not change it. */
export const COMPARISON_ALPHA = 0.05;
/** The variant that is the form as it stands, with no overrides. */
export const BASELINE_LABEL = "baseline";

export type AxisValue = number | string;

export interface ReplicatedComparisonFields {
  /** Dot-path of the field the variants change, e.g. training.learning_rate. */
  path: string;
  /** Comma-separated values for that field, besides the form's own. */
  rawValues: string;
  seedMode: ReplicatesSeedMode;
  nReplicates: number;
  rawSeeds: string;
  metric: string;
}

export interface ReplicatedComparisonPayload {
  variants: Record<string, Record<string, AxisValue>>;
  metric: string;
  n_replicates?: number;
  seeds?: number[];
}

/** A number when the token parses cleanly, else the trimmed string — the rule
 *  the sweep editor uses, so "0.001" and "resnet50" both land as typed. */
function coerce(token: string): AxisValue {
  const n = Number(token);
  return token !== "" && Number.isFinite(n) ? n : token;
}

/** Parse the comma-separated values; empty tokens are dropped. */
export function parseAxisValues(raw: string): AxisValue[] {
  return raw
    .split(",")
    .map((token) => token.trim())
    .filter((token) => token !== "")
    .map(coerce);
}

/** The variant's name: it is suffixed to the run name, so it becomes a folder
 *  name too and keeps to characters every file system takes. */
export function variantLabel(path: string, value: AxisValue): string {
  const leaf = path.trim().split(".").pop() ?? "";
  const shown = String(value).replace(/[^A-Za-z0-9._-]+/g, "-");
  return `${leaf}=${shown}`;
}

/** How many seeds the comparison trains each variant on. */
export function comparisonSeedCount(
  fields: Pick<ReplicatedComparisonFields, "seedMode" | "nReplicates" | "rawSeeds">,
): number {
  return fields.seedMode === "explicit"
    ? parseSeeds(fields.rawSeeds).length
    : fields.nReplicates;
}

/** How many variants the fields describe: the baseline plus each value. */
export function comparisonVariantCount(rawValues: string): number {
  return 1 + parseAxisValues(rawValues).length;
}

/** The smallest two-sided p Wilcoxon's signed-rank test can return with `n`
 *  pairs: `2 / 2**n`, the floor `significance.min_achievable_p` reports. */
export function wilcoxonFloor(n: number): number {
  if (n < 1) return 1;
  return Math.min(1, 2 ** (1 - n));
}

/** The floor when it sits above alpha (no difference could come out
 *  significant), else null. Below 8 pairs the backend always picks Wilcoxon,
 *  and it flags a comparison underpowered on exactly this condition. */
export function powerFloor(n: number, alpha: number = COMPARISON_ALPHA): number | null {
  const floor = wilcoxonFloor(n);
  return floor > alpha ? floor : null;
}

function axisProblem(t: Dict, path: string, values: AxisValue[]): string | null {
  const p = t.replicatedComparisonForm;
  if (path === "") return p.needPath;
  // The backend refuses it too (ADR-114): every variant would write to one place.
  if (path === "output" || path.startsWith("output.")) return p.outputPath;
  const variants = values.length + 1;
  if (variants < MIN_COMPARISON_VARIANTS) return p.needValue;
  if (variants > MAX_COMPARISON_VARIANTS) return p.tooManyValues;
  const labels = values.map((v) => variantLabel(path, v));
  if (new Set(labels).size !== labels.length) return p.duplicateValues;
  return null;
}

function seedProblem(t: Dict, fields: ReplicatedComparisonFields): string | null {
  if (fields.seedMode !== "explicit") return null;
  const seeds = parseSeeds(fields.rawSeeds);
  const problem = seedsProblem(t, seeds);
  if (problem) return problem;
  if (seeds.length > MAX_COMPARISON_SEEDS) {
    return t.replicatedComparisonForm.tooManySeeds(MAX_COMPARISON_SEEDS);
  }
  return null;
}

/** Shape the ReplicatedComparisonRequest body, or say why the fields cannot
 *  run. The baseline goes with no overrides: the backend applies each
 *  variant's overrides on top of the config the route receives. */
export function buildReplicatedComparisonPayload(
  t: Dict,
  fields: ReplicatedComparisonFields,
): { payload: ReplicatedComparisonPayload } | { problem: string } {
  const path = fields.path.trim();
  const values = parseAxisValues(fields.rawValues);
  const problem = axisProblem(t, path, values) ?? seedProblem(t, fields);
  if (problem) return { problem };

  const variants: Record<string, Record<string, AxisValue>> = { [BASELINE_LABEL]: {} };
  for (const value of values) variants[variantLabel(path, value)] = { [path]: value };
  const seeds =
    fields.seedMode === "explicit"
      ? { seeds: parseSeeds(fields.rawSeeds) }
      : { n_replicates: fields.nReplicates };
  return { payload: { variants, metric: fields.metric, ...seeds } };
}
