/** The request App sends for a paired comparison (ADR-061, ADR-193). Kept out
 *  of App so what reaches the server is testable without rendering the page. */

import { runReplicatedComparison } from "../api/client";
import type { RunResponse } from "../types/run";
import type { ReplicatedComparisonPayload } from "./replicated-comparison-form";

export interface ReplicatedComparisonLaunch {
  body: Record<string, unknown>;
  run: (payload: Record<string, unknown>) => Promise<RunResponse>;
  /** Trainings the job will run: variants × seeds, for the queue banner. */
  units: number;
}

/** Trainings a payload asks for: every variant over every seed. */
export function comparisonUnits(payload: ReplicatedComparisonPayload): number {
  const seeds = payload.seeds?.length ?? payload.n_replicates ?? 0;
  return Object.keys(payload.variants).length * seeds;
}

/** The body and the submit call for `task`'s replicated-comparison route.
 *
 *  Classification's comparison is a strategy of the GUI, not a block: each
 *  (variant, seed) trains as a single run, so the config goes with
 *  `block: "classification"`, as Réplicas does. */
export function replicatedComparisonLaunch(
  task: string,
  config: Record<string, unknown>,
  payload: ReplicatedComparisonPayload,
): ReplicatedComparisonLaunch {
  const base = task === "classification" ? { ...config, block: "classification" } : config;
  return {
    body: { config: base, ...payload },
    run: (p) => runReplicatedComparison(task, p),
    units: comparisonUnits(payload),
  };
}
