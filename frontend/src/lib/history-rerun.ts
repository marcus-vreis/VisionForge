/** Loading a run from History back into a task's form, as it trained or as a
 *  set of new seeds (ADR-156 amendment, #200). Kept out of App and the panels
 *  so the decisions (which tab, which seeds, which step) are testable without
 *  rendering the page. The config itself goes through the same import the
 *  panel's YAML button uses: this module only carries it there. */

import type { PanelStrategy } from "../components/ExperimentHeader";
import type { Dict } from "../i18n/pt";
import type { StepId } from "./steps";

/** "load" puts the config in the form as it trained; "replicates" also opens
 *  Réplicas with new seeds. */
export type RerunMode = "load" | "replicates";

/** How many seeds "Repetir com N seeds" asks for. */
export const RERUN_SEEDS = 3;

/** The seed a config trains with when it does not name one (`training.seed`). */
const DEFAULT_SEED = 42;

const BUILT_IN = ["regression", "segmentation", "anomaly", "detection"];
const CUSTOM_PREFIX = "custom:";

/** What History knows about a run, as `GET /api/runs/{id}` serves it. */
export interface RerunSource {
  run_id: string;
  /** `custom:<key>` for a researcher's task, else a built-in name. */
  task?: string;
  config: Record<string, unknown>;
  /** On a group: what kind of job it was and the seeds it asked for. */
  group?: { kind: string; seeds: number[] } | null;
}

/** What App receives from History. */
export interface RerunRequest {
  family: string;
  config: Record<string, unknown>;
  mode: RerunMode;
  seeds: number[];
  source: string;
}

/** What App keeps per task until the task's panel has taken it. */
export interface PendingImport {
  /** Tells two loads of the same run apart. */
  id: number;
  config: Record<string, unknown>;
  mode: RerunMode;
  seeds: number[];
  /** Named where the form says what it imported. */
  source: string;
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

/** The task family of a run (the backend's rule): the server's own `task`
 *  when it sent one, else the config's built-in name, else classification,
 *  whose config carries no `task`. */
export function runFamily(source: Pick<RerunSource, "task" | "config">): string {
  if (source.task) return source.task;
  const task = isRecord(source.config) ? source.config["task"] : undefined;
  return typeof task === "string" && BUILT_IN.includes(task) ? task : "classification";
}

/** The tab a family opens: a researcher's task is its own key. */
export function tabForFamily(family: string): string {
  return family.startsWith(CUSTOM_PREFIX) ? family.slice(CUSTOM_PREFIX.length) : family;
}

/** The seeds the run already used: a group's, or the config's own. */
export function usedSeeds(source: Pick<RerunSource, "config" | "group">): number[] {
  if (source.group && source.group.seeds.length > 0) return [...source.group.seeds];
  const training = isRecord(source.config) ? source.config["training"] : undefined;
  const seed = isRecord(training) ? training["seed"] : undefined;
  return [typeof seed === "number" && Number.isInteger(seed) && seed >= 0 ? seed : DEFAULT_SEED];
}

/** `n` consecutive seeds after the largest one used, so none repeats a
 *  training that already exists. */
export function freshSeeds(used: readonly number[], n = RERUN_SEEDS): number[] {
  const start = used.length > 0 ? Math.max(...used) + 1 : DEFAULT_SEED;
  return Array.from({ length: n }, (_, i) => start + i);
}

/** What the History actions may do with a run. */
export interface RerunOptions {
  /** Why both actions are off, shown beside them; null when they work. */
  disabledReason: string | null;
  /** "Repetir com N seeds" is offered: a single run or a replicate set. A
   *  sweep, a K-fold or a comparison has no one config to repeat. */
  canReplicate: boolean;
  /** The seeds Réplicas would open with. */
  seeds: number[];
}

/** Whether the run can go back to a form here. `families` are the task
 *  families that have a tab now: a researcher's task that was removed has none. */
export function rerunOptions(
  t: Dict,
  source: RerunSource,
  families: readonly string[],
): RerunOptions {
  const family = runFamily(source);
  const seeds = freshSeeds(usedSeeds(source));
  const canReplicate = !source.group || source.group.kind === "replicates";
  const r = t.runDetail.rerun;
  let disabledReason: string | null = null;
  if (!isRecord(source.config) || Object.keys(source.config).length === 0) {
    disabledReason = r.noConfig;
  } else if (!families.includes(family)) {
    disabledReason = family.startsWith(CUSTOM_PREFIX)
      ? r.customRemoved(tabForFamily(family))
      : r.noTab(family);
  }
  return { disabledReason, canReplicate, seeds };
}

/** The request one of the actions sends, or null when that action is off. */
export function rerunRequest(
  t: Dict,
  source: RerunSource,
  families: readonly string[],
  mode: RerunMode,
): RerunRequest | null {
  const options = rerunOptions(t, source, families);
  if (options.disabledReason !== null) return null;
  if (mode === "replicates" && !options.canReplicate) return null;
  return {
    family: runFamily(source),
    config: source.config,
    mode,
    seeds: mode === "replicates" ? options.seeds : [],
    source: source.run_id,
  };
}

/** Where App sends a request: the tab to open and what that tab's panel takes
 *  once it is ready. Null when no tab holds the family any more. */
export function planRerun(
  request: RerunRequest,
  families: readonly string[],
  id: number,
): { tab: string; pending: PendingImport } | null {
  if (!families.includes(request.family)) return null;
  return {
    tab: tabForFamily(request.family),
    pending: {
      id,
      config: request.config,
      mode: request.mode,
      seeds: request.seeds,
      source: request.source,
    },
  };
}

/** App's per-task queue once a panel has applied its import. A newer request
 *  for the same tab (another id) stays. */
export function takePendingImport(
  pending: Readonly<Record<string, PendingImport>>,
  tab: string,
  id: number,
): Readonly<Record<string, PendingImport>> {
  if (pending[tab]?.id !== id) return pending;
  const rest = { ...pending };
  delete rest[tab];
  return rest;
}

/** What the panel does once the import ran. */
export interface ImportOutcome {
  /** The strategy to select; null leaves the form's as it was. */
  strategy: PanelStrategy | null;
  /** The step to show: Revisão holds the import's message, Estratégia the
   *  Réplicas card. */
  step: StepId;
  /** Seeds for the Réplicas card, when it opens. */
  seeds: number[] | null;
}

/** `problem` is the import's message; `refused` says the panel did not load
 *  the config because of it (the standalone panels refuse a file with
 *  problems, classification loads what it can and warns). */
export function importOutcome(
  pending: Pick<PendingImport, "mode" | "seeds">,
  problem: string | null,
  refused: boolean,
): ImportOutcome {
  if (problem !== null && refused) return { strategy: null, step: "review", seeds: null };
  if (pending.mode === "replicates") {
    return {
      strategy: "replicates",
      step: problem === null ? "strategy" : "review",
      seeds: pending.seeds,
    };
  }
  return { strategy: "simple", step: "review", seeds: null };
}
