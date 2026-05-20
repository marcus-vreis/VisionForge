/** Monospaced label with an optional accent dot and trailing hint. */
export function FieldLabel({
  children,
  hint,
  dot = false,
}: {
  children: React.ReactNode;
  hint?: string;
  dot?: boolean;
}) {
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
      <span>{children}</span>
      {hint && (
        <span
          style={{
            marginLeft: "auto",
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
