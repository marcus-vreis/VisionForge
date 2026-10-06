import { Eye, EyeOff } from "lucide-react";
import { useState } from "react";
import { useT } from "../../i18n/useT";
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
  /** A credential: masked while typing (with a reveal toggle), never offered to
   * the browser's autofill. Opt-in, so every other field stays plain text. */
  secret?: boolean;
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
  secret = false,
}: TextFieldProps) {
  const t = useT();
  const [revealed, setRevealed] = useState(false);
  const masked = secret && !revealed;

  return (
    <div>
      <FieldLabel dot hint={hint} help={help}>
        {label}
      </FieldLabel>
      <div style={shellStyle}>
        <input
          type={masked ? "password" : "text"}
          // `off` is ignored by browsers for password inputs, which still offer
          // to fill a saved *login* here; `new-password` is the value that stops it.
          autoComplete={secret ? "new-password" : undefined}
          spellCheck={secret ? false : undefined}
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
        {secret && (
          <button
            type="button"
            onClick={() => setRevealed((r) => !r)}
            aria-label={revealed ? t.textField.hideSecret : t.textField.showSecret}
            title={revealed ? t.textField.hideSecret : t.textField.showSecret}
            style={{
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              padding: "0 12px",
              alignSelf: "stretch",
              background: "transparent",
              border: "none",
              color: "var(--vf-text-dim)",
              cursor: "pointer",
            }}
          >
            {revealed ? <EyeOff size={15} aria-hidden /> : <Eye size={15} aria-hidden />}
          </button>
        )}
      </div>
    </div>
  );
}
