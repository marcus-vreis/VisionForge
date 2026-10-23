/** Render validation feedback at its field's stable description target. */
export function FieldError({ id, error }: { id?: string; error?: string }) {
  return error ? (
    <p id={id} style={{ fontSize: 11, color: "oklch(0.704 0.191 22.216)", marginTop: 4, fontFamily: "var(--font-mono)" }}>
      {error}
    </p>
  ) : null;
}
