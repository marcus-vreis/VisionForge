import { useT } from "../i18n/useT";
import { reasonRows } from "../lib/dataset-coco";
import {
  button,
  ghostButton,
  mono,
  sectionLabel,
} from "./datasetExplorerStyles";
import { TextField } from "./controls";
import { Progress } from "./JobProgress";

/** The running job of a conversion: what it is doing, its bar and the cancel button. */
export function CocoJobBar({
  text,
  pct,
  cancelling,
  onCancel,
}: {
  text: string;
  pct: number;
  cancelling: boolean;
  onCancel: () => void;
}) {
  const t = useT().datasetCoco;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={mono}>{text}</div>
      <Progress pct={pct} />
      <div>
        <button
          type="button"
          style={ghostButton}
          disabled={cancelling}
          onClick={onCancel}
        >
          {cancelling ? t.cancelling : t.cancel}
        </button>
      </div>
    </div>
  );
}

/** Reasons with their counts, in a sentence per reason. `labels` translates the codes. */
export function ReasonSummary({
  counts,
  labels,
}: {
  counts: Record<string, number>;
  labels: Record<string, string>;
}) {
  const t = useT().datasetCoco;
  const rows = reasonRows(counts);
  if (rows.length === 0) return null;
  return (
    <div style={mono}>
      {rows
        .map((r) => t.reasonCount(labels[r.reason] ?? r.reason, r.count))
        .join("  ·  ")}
    </div>
  );
}

/** The destination name and the button that starts the write. */
export function DestinationRow({
  name,
  onName,
  ok,
  canApply,
  applyLabel,
  problem,
  onApply,
}: {
  name: string;
  onName: (v: string) => void;
  ok: boolean;
  canApply: boolean;
  applyLabel: string;
  /** What the server said about the name (taken, invalid), if anything. */
  problem: string | null;
  onApply: () => void;
}) {
  const t = useT().datasetCoco;
  const error = name !== "" && !ok ? t.destInvalid : (problem ?? undefined);
  return (
    <>
      <div style={sectionLabel}>{t.destTitle}</div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 12 }}>
        <div style={{ flex: 1, maxWidth: 420 }}>
          <TextField
            label={t.destName}
            value={name}
            onChange={onName}
            hint={t.destHint}
            mono
            error={error}
          />
        </div>
        <button
          type="button"
          style={button}
          disabled={!canApply}
          onClick={onApply}
        >
          {applyLabel}
        </button>
      </div>
    </>
  );
}
