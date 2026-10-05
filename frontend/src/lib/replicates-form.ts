/** Pure helpers for the ReplicatesCard (ADR-056/059) — parsing the explicit
 *  seed list and shaping the backend request. Kept out of the component so the
 *  validation rules are unit-testable. */

import type { Dict } from "../i18n/pt";

export type ReplicatesSeedMode = "auto" | "explicit";

export interface ReplicatesPayload {
  metric: string;
  n_replicates?: number;
  seeds?: number[];
}

/** Parse a comma-separated seed list into integers; non-numeric tokens are
 *  dropped, fractional values are floored to ints. */
export function parseSeeds(raw: string): number[] {
  return raw
    .split(",")
    .map((t) => t.trim())
    .filter((t) => t !== "")
    .map((t) => Number(t))
    .filter((n) => Number.isFinite(n) && n >= 0)
    .map((n) => Math.floor(n));
}

/** Explicit seeds are valid when there are ≥2 and no duplicates — mirrors the
 *  backend's 422 rules so the button can disable before a doomed submit. The
 *  problem is worded in the language of `t`. */
export function seedsProblem(t: Dict, seeds: number[]): string | null {
  if (seeds.length < 2) return t.replicatesForm.needTwoSeeds;
  if (new Set(seeds).size !== seeds.length) return t.replicatesForm.duplicateSeeds;
  return null;
}

/** Shape the ReplicatesRequest body: explicit seeds win; otherwise the backend
 *  derives n consecutive seeds from the config's own training.seed. */
export function buildReplicatesPayload(
  mode: ReplicatesSeedMode,
  nReplicates: number,
  rawSeeds: string,
  metric: string,
): ReplicatesPayload {
  if (mode === "explicit") {
    return { metric, seeds: parseSeeds(rawSeeds) };
  }
  return { metric, n_replicates: nReplicates };
}
