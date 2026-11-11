import type { CSSProperties } from "react";
import { useT } from "../i18n/useT";
import {
  rerunOptions,
  rerunRequest,
  type RerunMode,
  type RerunRequest,
  type RerunSource,
} from "../lib/history-rerun";

/** What History needs from App to send a run back to a form (#200). */
export interface RerunHandle {
  /** The task families that have a tab now (`custom:<key>` for a researcher's). */
  families: readonly string[];
  onRerun: (request: RerunRequest) => void;
}

const buttonStyle: CSSProperties = {
  padding: "6px 12px",
  background: "rgb(var(--lt-ink, 255 255 255) / 0.04)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 8,
  color: "var(--vf-text-dim)",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  letterSpacing: "0.10em",
  textTransform: "uppercase",
  cursor: "pointer",
};

const offStyle: CSSProperties = { ...buttonStyle, cursor: "not-allowed", opacity: 0.5 };

/** "Carregar no formulário" and "Repetir com N seeds" for one run or group.
 *  When the run cannot go back to a form, both stay on screen, off, with the
 *  reason written beside them. */
export function RerunActions({ source, handle }: { source: RerunSource; handle: RerunHandle }) {
  const t = useT();
  const r = t.runDetail.rerun;
  const options = rerunOptions(t, source, handle.families);
  const off = options.disabledReason !== null;
  const send = (mode: RerunMode) => {
    const request = rerunRequest(t, source, handle.families, mode);
    if (request) handle.onRerun(request);
  };

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
      <button
        type="button"
        onClick={() => send("load")}
        disabled={off}
        title={options.disabledReason ?? r.loadTitle}
        style={off ? offStyle : buttonStyle}
      >
        {r.load}
      </button>
      {options.canReplicate && (
        <button
          type="button"
          onClick={() => send("replicates")}
          disabled={off}
          title={options.disabledReason ?? r.replicatesTitle(options.seeds.join(", "))}
          style={off ? offStyle : buttonStyle}
        >
          {r.replicates(options.seeds.length)}
        </button>
      )}
      {off && (
        <span
          role="note"
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
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
