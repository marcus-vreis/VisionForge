import { Repeat, RotateCcw } from "lucide-react";
import { useT } from "../i18n/useT";
import {
  rerunOptions,
  rerunRequest,
  runFamily,
  type RerunMode,
  type RerunRequest,
  type RerunSource,
} from "../lib/history-rerun";
import { routeOpen, strategyRoute, useSession } from "../lib/session";
import { ActionButton } from "./ui/ActionButton";

/** What History needs from App to send a run back to a form (#200). */
export interface RerunHandle {
  /** The task families that have a tab now (`custom:<key>` for a researcher's). */
  families: readonly string[];
  onRerun: (request: RerunRequest) => void;
}

/** "Carregar no formulário" and "Repetir com N seeds" for one run or group.
 *  When the run cannot go back to a form, both stay on screen, off, with the
 *  reason written beside them. */
export function RerunActions({ source, handle }: { source: RerunSource; handle: RerunHandle }) {
  const t = useT();
  const r = t.runDetail.rerun;
  const options = rerunOptions(t, source, handle.families);
  const off = options.disabledReason !== null;
  // A shared server may not run replicates yet (ADR-216).
  const replicatesOpen = routeOpen(
    useSession(),
    strategyRoute(runFamily(source), "replicates"),
  );
  const send = (mode: RerunMode) => {
    const request = rerunRequest(t, source, handle.families, mode);
    if (request) handle.onRerun(request);
  };

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
      <ActionButton
        icon={<RotateCcw />}
        onClick={() => send("load")}
        disabled={off}
        title={options.disabledReason ?? r.loadTitle}
      >
        {r.load}
      </ActionButton>
      {options.canReplicate && replicatesOpen && (
        <ActionButton
          icon={<Repeat />}
          onClick={() => send("replicates")}
          disabled={off}
          title={options.disabledReason ?? r.replicatesTitle(options.seeds.join(", "))}
        >
          {r.replicates(options.seeds.length)}
        </ActionButton>
      )}
      {off && (
        <span
          role="note"
          style={{
            fontSize: 13,
            lineHeight: 1.5,
            color: "var(--vf-text-muted)",
          }}
        >
          {options.disabledReason}
        </span>
      )}
    </div>
  );
}
