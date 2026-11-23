/**
 * A saved run's per-epoch history under today's names (ADR-241).
 *
 * Anomaly scores every epoch on the test split, and wrote that AUROC as
 * `val_auroc` until it was renamed `test_auroc`. A run saved before the rename
 * still says `val_auroc`; the run detail is passed through here as it arrives
 * (`fetchRunDetail`), so every chart, comparison and story reads one name. This
 * is the one place the old name is known in the page.
 *
 * Only anomaly runs are renamed: a researcher's own task (ADR-058) streams
 * `val_<metric>` for a metric it measured on a validation split, so its
 * `val_auroc` means what it says.
 */

/** Old history key -> current key, for anomaly runs. */
const ANOMALY_RENAMED: Readonly<Record<string, string>> = { val_auroc: "test_auroc" };

/** One anomaly epoch record under the current names; the input is untouched.
 *  A record that already carries the new name keeps its value. */
export function upgradeAnomalyEpoch<T extends Record<string, unknown>>(record: T): T {
  const out: Record<string, unknown> = { ...record };
  for (const [old, current] of Object.entries(ANOMALY_RENAMED)) {
    if (!(old in out)) continue;
    const value = out[old];
    delete out[old];
    if (!(current in out)) out[current] = value;
  }
  return out as T;
}

interface WithHistory {
  task?: string;
  config: Record<string, unknown>;
  history: ReadonlyArray<Record<string, unknown>>;
}

/** The task as the server names it, else the built-in name in `config.task`. */
function isAnomaly(run: { task?: string; config: Record<string, unknown> }): boolean {
  if (typeof run.task === "string" && run.task !== "") return run.task === "anomaly";
  return run.config?.task === "anomaly";
}

/** The run with its history under the current names: the same object when there
 *  is nothing to rename, else a copy with a new history. */
export function upgradeRunHistory<R extends WithHistory>(run: R): R {
  if (!Array.isArray(run.history) || !isAnomaly(run)) return run;
  return { ...run, history: run.history.map((row) => upgradeAnomalyEpoch(row)) };
}
