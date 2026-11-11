import { useEffect, useRef, useState } from "react";
import type { PanelStrategy } from "../components/ExperimentHeader";
import { importOutcome, type PendingImport } from "../lib/history-rerun";
import { useStepNav } from "../lib/steps";

/** The props App gives every task panel for a config sent from History (#200). */
export interface PendingImportProps {
  /** A run's config History sent to this task; applied once the panel is ready. */
  pendingImport?: PendingImport | null;
  /** The panel applied the pending import with this id. */
  onImportApplied?: (id: number) => void;
}

/** What the Revisão name card shows about an import that did not come from
 *  its own file button. */
export interface ImportNotice {
  id: number;
  /** Named in the "imported" line. */
  source: string;
  /** The import's message, null when it went through clean. */
  problem: string | null;
}

/** Seeds the Réplicas card opens on. */
export interface ReplicatesPreset {
  id: number;
  seeds: number[];
}

interface Options extends PendingImportProps {
  taskKey: string;
  /** The panel can import: its schema and form are in place. */
  ready: boolean;
  /** The panel's YAML import, the very handler its button calls. */
  importConfig: (data: Record<string, unknown>) => string | null | Promise<string | null>;
  /** The import refuses a config with problems instead of loading what it can. */
  refusesOnProblem: boolean;
  onStrategy: (strategy: PanelStrategy) => void;
}

/** Applies a pending History import once: through the panel's own YAML import,
 *  then the strategy and step `importOutcome` decides. Returns what the name
 *  card and the Réplicas card show. */
export function usePendingImport(options: Options): {
  notice: ImportNotice | null;
  replicatesPreset: ReplicatesPreset | null;
} {
  const nav = useStepNav();
  const [notice, setNotice] = useState<ImportNotice | null>(null);
  const [replicatesPreset, setReplicatesPreset] = useState<ReplicatesPreset | null>(null);
  // The handlers close over the form; the effect reads the latest ones.
  const latest = useRef(options);
  useEffect(() => {
    latest.current = options;
  });
  const taken = useRef<number | null>(null);
  const { pendingImport, ready } = options;

  useEffect(() => {
    if (!pendingImport || !ready || taken.current === pendingImport.id) return;
    taken.current = pendingImport.id;
    const current = latest.current;
    void Promise.resolve(current.importConfig(pendingImport.config))
      .then((problem) => {
        const outcome = importOutcome(pendingImport, problem, current.refusesOnProblem);
        if (outcome.strategy) current.onStrategy(outcome.strategy);
        if (outcome.seeds) setReplicatesPreset({ id: pendingImport.id, seeds: outcome.seeds });
        setNotice({ id: pendingImport.id, source: pendingImport.source, problem });
        nav?.setStep(current.taskKey, outcome.step);
      })
      .finally(() => current.onImportApplied?.(pendingImport.id));
  }, [pendingImport, ready, nav]);

  return { notice, replicatesPreset };
}
