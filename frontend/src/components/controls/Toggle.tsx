import { FieldLabel } from "./FieldLabel";

interface ToggleProps {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
  hint?: string;
}

/** Full-width toggle button with an inline track indicator. */
export function Toggle({ label, value, onChange, hint }: ToggleProps) {
  return (
    <div>
      <FieldLabel dot hint={hint}>
        {label}
      </FieldLabel>
      <button
        type="button"
        onClick={() => onChange(!value)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          width: "100%",
          padding: "11px 14px",
          background: "rgba(12,14,18,0.65)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 10,
          cursor: "pointer",
        }}
      >
        <span
          style={{
            position: "relative",
            width: 32,
            height: 18,
            borderRadius: 999,
            background: value ? "var(--accent-vf)" : "rgba(255,255,255,0.12)",
            boxShadow: value ? "0 0 10px var(--accent-glow)" : "none",
            transition: "all 200ms ease",
            flexShrink: 0,
          }}
        >
          <span
            style={{
              position: "absolute",
              top: 2,
              left: value ? 16 : 2,
              width: 14,
              height: 14,
              borderRadius: "50%",
              background: "#fff",
              transition: "left 200ms ease",
            }}
          />
        </span>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            color: value ? "var(--vf-text)" : "var(--vf-text-dim)",
            letterSpacing: "0.06em",
            textTransform: "uppercase",
          }}
        >
          {value ? "ATIVADO" : "DESATIVADO"}
        </span>
      </button>
    </div>
  );
}
