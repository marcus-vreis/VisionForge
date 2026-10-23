import { FieldLabel } from "./FieldLabel";
import { FieldError } from "./FieldError";
import { useFieldAccessibility } from "./useFieldAccessibility";

interface SegmentedOption {
  value: string;
  label: string;
}

interface SegmentedProps {
  error?: string;
  /** Explanation shown in the label info dot. */
  help?: string;
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
  help,
  error,
}: SegmentedProps) {
  const field = useFieldAccessibility({ hint, help, error });
  const normalize = (opt: string | SegmentedOption): SegmentedOption =>
    typeof opt === "string" ? { value: opt, label: opt } : opt;

  return (
    <div>
      <FieldLabel dot id={field.labelId} hintId={field.hintId} helpId={field.helpId} hint={hint} help={help}>
        {label}
      </FieldLabel>
      <div
        role="group"
        aria-labelledby={field.labelId}
        aria-describedby={field["aria-describedby"]}
        aria-invalid={field["aria-invalid"]}
        style={{
          display: "flex",
          background: "rgb(var(--lt-surface, 12 14 18) / 0.65)",
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
              aria-pressed={active}
              title={o.label}
              onClick={() => onChange(o.value)}
              style={{
                flex: 1,
                padding: "8px 6px",
                border: "none",
                borderRadius: 7,
                background: active ? "var(--accent-soft)" : "transparent",
                color: active ? "var(--vf-text)" : "var(--vf-text-dim)",
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                letterSpacing: "0.04em",
                // One line per option: a wrapped label made this control taller
                // than the fields beside it.
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
                minWidth: 0,
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
      <FieldError id={field.errorId} error={error} />
    </div>
  );
}
