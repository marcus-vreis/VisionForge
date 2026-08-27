import { FieldLabel } from "./FieldLabel";

const shellStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  background: "rgba(12,14,18,0.65)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 10,
  transition: "border-color 160ms ease, box-shadow 160ms ease",
};

interface TextFieldProps {
  /** Explanation shown in the label info dot. */
  help?: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  hint?: string;
  mono?: boolean;
}

/** Text input field with accent dot label. */
export function TextField({
  label,
  value,
  onChange,
  placeholder,
  hint,
  mono = false,
  help,
}: TextFieldProps) {
  return (
    <div>
      <FieldLabel dot hint={hint} help={help}>
        {label}
      </FieldLabel>
      <div style={shellStyle}>
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          style={{
            flex: 1,
            background: "transparent",
            border: "none",
            outline: "none",
            padding: "12px 14px",
            fontFamily: mono ? "var(--font-mono)" : "var(--font-sans)",
            fontSize: 13,
            color: "var(--vf-text)",
            letterSpacing: "0.01em",
            width: "100%",
          }}
        />
      </div>
    </div>
  );
}
