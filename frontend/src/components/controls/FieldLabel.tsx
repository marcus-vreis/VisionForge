import { InfoDot } from "./InfoDot";

/** Monospaced label with an optional accent dot, info dot and trailing hint. */
export function FieldLabel({
  children,
  hint,
  dot = false,
  help,
  htmlFor,
  id,
  hintId,
  helpId,
}: {
  children: React.ReactNode;
  hint?: string;
  dot?: boolean;
  /** Explanation for this field, shown in an info dot beside the label. */
  help?: string;
  htmlFor?: string;
  id?: string;
  hintId?: string;
  helpId?: string;
}) {
  const Label = htmlFor ? "label" : "span";
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        marginBottom: 8,
        fontSize: 11,
        letterSpacing: "0.14em",
        textTransform: "uppercase",
        color: "var(--vf-text-dim)",
        fontFamily: "var(--font-mono)",
      }}
    >
      {dot && (
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: "var(--accent-vf)",
            boxShadow: "0 0 8px var(--accent-glow)",
            flexShrink: 0,
          }}
        />
      )}
      {/* The label never wraps: a hint beside it squeezed "NOME DO / EXPERIMENTO"
          onto two lines, which pushed that field below its neighbours. The hint
          gives way instead, cut with an ellipsis and whole in its tooltip. */}
      <Label id={id} htmlFor={htmlFor} style={{ whiteSpace: "nowrap", flexShrink: 0 }}>{children}</Label>
      {help && <InfoDot text={help} />}
      {help && helpId && <span id={helpId} className="sr-only">{help}</span>}
      {hint && (
        <span
          id={hintId}
          title={hint}
          style={{
            marginLeft: "auto",
            minWidth: 0,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            color: "var(--vf-text-muted)",
            textTransform: "none",
            letterSpacing: 0,
            fontSize: 11,
          }}
        >
          {hint}
        </span>
      )}
    </div>
  );
}
