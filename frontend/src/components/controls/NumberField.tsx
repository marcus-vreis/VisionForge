import { FieldLabel } from "./FieldLabel";

const shellStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  background: "rgba(12,14,18,0.65)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 10,
  transition: "border-color 160ms ease, box-shadow 160ms ease",
};

const inputStyle: React.CSSProperties = {
  flex: 1,
  background: "transparent",
  border: "none",
  outline: "none",
  padding: "12px 14px",
  fontFamily: "var(--font-mono)",
  fontSize: 13,
  color: "var(--vf-text)",
  letterSpacing: "0.01em",
  width: "100%",
};

interface NumberFieldProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  hint?: string;
  suffix?: string;
}

/** Numeric input field with accent dot label. */
export function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  hint,
  suffix,
}: NumberFieldProps) {
  const onBlur = (e: React.FocusEvent<HTMLInputElement>) => {
    let v = parseFloat(e.target.value);
    if (isNaN(v)) v = value;
    if (typeof min === "number") v = Math.max(min, v);
    if (typeof max === "number") v = Math.min(max, v);
    onChange(v);
  };

  return (
    <div>
      <FieldLabel dot hint={hint}>
        {label}
      </FieldLabel>
      <div style={shellStyle}>
        <input
          type="text"
          inputMode="decimal"
          defaultValue={value}
          key={value}
          onBlur={onBlur}
          onKeyDown={(e) => {
            if (e.key === "Enter") (e.target as HTMLInputElement).blur();
          }}
          style={inputStyle}
          step={step}
        />
        {suffix && (
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 12,
              color: "var(--vf-text-muted)",
              paddingRight: 14,
            }}
          >
            {suffix}
          </span>
        )}
      </div>
    </div>
  );
}
