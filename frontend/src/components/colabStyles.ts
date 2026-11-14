/** Shared styles of the Colab panel and its Drive folder picker. */
import type { CSSProperties } from "react";

export const MONO: CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  letterSpacing: "0.08em",
};

/** Body copy: one short line per control. */
export const BODY: CSSProperties = {
  fontSize: 13,
  lineHeight: 1.5,
  color: "var(--vf-text)",
  margin: 0,
};

export const DIM: CSSProperties = { ...BODY, color: "var(--vf-text-dim)" };

export const ERROR: CSSProperties = {
  ...BODY,
  color: "oklch(calc(0.75 - var(--lt-flip, 0) * 0.2) 0.15 25)",
};

/** A field's label, mono and small caps like the form's `FieldLabel`. */
export const LABEL: CSSProperties = {
  ...MONO,
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
};

export const LINK: CSSProperties = {
  ...MONO,
  fontSize: 12,
  letterSpacing: "0.04em",
  background: "none",
  border: "none",
  padding: 0,
  color: "var(--vf-text-dim)",
  textDecoration: "underline",
  textUnderlineOffset: 3,
  cursor: "pointer",
};

export const ACTION: CSSProperties = {
  ...MONO,
  fontSize: 12,
  padding: "9px 14px",
  borderRadius: 8,
  border: "1px solid var(--accent-vf)",
  background: "rgb(var(--lt-ink, 255 255 255) / 0.04)",
  color: "var(--vf-text)",
  cursor: "pointer",
  textTransform: "uppercase",
  whiteSpace: "nowrap",
};

export const FIELD: CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 13,
  padding: "9px 11px",
  borderRadius: 8,
  border: "1px solid var(--vf-panel-stroke)",
  background: "rgb(var(--lt-shade, 0 0 0) / 0.35)",
  color: "var(--vf-text)",
  minWidth: 0,
};

export const CHECK_ROW: CSSProperties = {
  ...BODY,
  display: "flex",
  gap: 10,
  alignItems: "center",
  cursor: "pointer",
};

export const STACK: CSSProperties = { display: "flex", flexDirection: "column", gap: 8 };
