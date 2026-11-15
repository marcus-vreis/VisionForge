/** The request App sends for a multi-seed replicate set (ADR-056). Kept out of
 *  App so what reaches the server is testable without rendering the page. */

import { runReplicates } from "../api/client";
import type { RunResponse } from "../types/run";
import type { ReplicatesPayload } from "./replicates-form";

export interface ReplicatesLaunch {
  body: Record<string, unknown>;
  run: (payload: Record<string, unknown>) => Promise<RunResponse>;
}

/** The body and the submit call for `task`'s replicate route.
 *
 *  Classification's Réplicas is a strategy of the GUI, not a block: the
 *  replicate runner trains each seed as a single run, so the config always
 *  goes with `block: "classification"`. */
export function replicatesLaunch(
  task: string,
  config: Record<string, unknown>,
  payload: ReplicatesPayload,
): ReplicatesLaunch {
  const base = task === "classification" ? { ...config, block: "classification" } : config;
  return {
    body: { config: base, ...payload },
    run: (p) => runReplicates(task, p),
  };
}
