import { FieldLabel } from "./FieldLabel";

interface SegmentedOption {
  value: string;
  label: string;
}

interface SegmentedProps {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: (string | SegmentedOption)[];
  hint?: string;
}

/** Segmented button group for small discrete choices. */
export function Segmented({
  label,
  value,
  onChange,
  options,
  hint,
}: SegmentedProps) {
  const normalize = (opt: string | SegmentedOption): SegmentedOption =>
    typeof opt === "string" ? { value: opt, label: opt } : opt;

  return (
    <div>
      <FieldLabel dot hint={hint}>
        {label}
      </FieldLabel>
      <div
        style={{
          display: "flex",
          background: "rgba(12,14,18,0.65)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 10,
          padding: 4,
          gap: 2,
        }}
      >
        {options.map((opt) => {
          const o = normalize(opt);
          const active = o.value === value;
          return (
            <button
              key={o.value}
              type="button"
              onClick={() => onChange(o.value)}
              style={{
                flex: 1,
                padding: "8px 10px",
                border: "none",
                borderRadius: 7,
                background: active ? "var(--accent-soft)" : "transparent",
                color: active ? "var(--vf-text)" : "var(--vf-text-dim)",
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                letterSpacing: "0.04em",
                transition: "all 160ms ease",
                position: "relative",
                cursor: "pointer",
              }}
            >
              {active && (
                <span
                  style={{
                    position: "absolute",
                    inset: 0,
                    borderRadius: 7,
                    border: "1px solid var(--accent-vf)",
                    boxShadow: "inset 0 0 12px var(--accent-glow)",
                    pointerEvents: "none",
                  }}
                />
              )}
              {o.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
