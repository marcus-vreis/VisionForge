import { InfoDot } from "./InfoDot";

// The accent dot (6 px) and the info dot (14 px), each with the row's 8 px gap.
const ACCENT_DOT_SPACE = 6 + 8;
const INFO_DOT_SPACE = 14 + 8;

/** The width the dots on the label's row take, in px. */
function besideLabel(dot: boolean, help: string | undefined): number {
  return (dot ? ACCENT_DOT_SPACE : 0) + (help ? INFO_DOT_SPACE : 0);
}

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
      className="vf-field-label"
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        minWidth: 0,
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
          gives way instead, cut with an ellipsis and whole in its tooltip. At
          phone width there is no tooltip to hover, so the hint takes a line of
          its own under the label (.vf-field-hint in index.css). A label longer
          than its whole field ("GRAD-CAM AO VIVO: IMAGENS POR CLASSE") is the one
          case that wraps: on one line it ran past the field and the card. Its
          width leaves room for the dots on the same row, or they would. */}
      <Label
        id={id}
        htmlFor={htmlFor}
        style={{ flexShrink: 0, maxWidth: `calc(100% - ${besideLabel(dot, help)}px)` }}
      >
        {children}
      </Label>
      {help && <InfoDot text={help} />}
      {help && helpId && <span id={helpId} className="sr-only">{help}</span>}
      {hint && (
        <span
          id={hintId}
          title={hint}
          className="vf-field-hint"
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
