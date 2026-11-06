import type { CSSProperties } from "react";

/** Inline styles shared by the dataset explorer and the incorporate panel. */

export const sectionLabel: CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  letterSpacing: "0.22em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
};

export const card: CSSProperties = {
  padding: 16,
  background: "rgb(var(--lt-ink, 255 255 255) / 0.02)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 12,
  display: "flex",
  flexDirection: "column",
  gap: 12,
};

export const mono: CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  color: "var(--vf-text-dim)",
  lineHeight: 1.6,
};

export const button: CSSProperties = {
  padding: "8px 14px",
  background: "var(--accent-soft)",
  border: "1px solid var(--accent-vf)",
  borderRadius: 10,
  color: "var(--vf-text)",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  letterSpacing: "0.10em",
  textTransform: "uppercase",
  cursor: "pointer",
  whiteSpace: "nowrap",
};

export const ghostButton: CSSProperties = {
  ...button,
  background: "transparent",
  border: "1px solid var(--vf-panel-stroke)",
  color: "var(--vf-text-dim)",
};

export const danger: CSSProperties = { ...mono, color: "var(--vf-danger, #f87171)" };
