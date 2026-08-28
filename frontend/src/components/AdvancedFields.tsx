import { useState } from "react";
import { Chevron } from "./controls";

/** Holds the knobs that get set once and then left alone.
 *
 * The classification into basic and advanced has existed in `param-help.ts`
 * since the complaint "conheço apenas metade desses; tá bem difícil de
 * entender, muita coisa junta" — but nothing ever read it, so every field was
 * shown at once regardless. Splitting by *how often a value changes* puts
 * epochs and learning rate in front and the optimizer behind a click, even
 * though the optimizer matters enormously: it gets decided once.
 *
 * It opens by itself when any field inside differs from its default. A section
 * that hides a value the researcher deliberately set would be worse than one
 * that shows everything.
 */
export function AdvancedFields({
  count,
  startOpen = false,
  children,
}: {
  count: number;
  startOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(startOpen);

  if (count === 0) return null;

  return (
    <div style={{ marginBottom: 26 }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "8px 12px",
          width: "100%",
          background: "rgba(255,255,255,0.02)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 8,
          color: "var(--vf-text-dim)",
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.16em",
          textTransform: "uppercase",
          cursor: "pointer",
        }}
      >
        <Chevron open={open} size={10} />
        <span>avançado</span>
        <span style={{ color: "var(--vf-text-muted)" }}>({count})</span>
        {startOpen && !open && (
          <span style={{ marginLeft: "auto", textTransform: "none", letterSpacing: 0 }}>
            valores alterados
          </span>
        )}
      </button>
      {open && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: 18,
            marginTop: 18,
          }}
        >
          {children}
        </div>
      )}
    </div>
  );
}
