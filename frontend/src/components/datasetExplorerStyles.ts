import type { CSSProperties } from "react";

/** Inline styles shared by the dataset lab and its tools. Prose is `text`
 *  (13px, sans); `mono` is for data: counts, paths, table cells. */

export const sectionLabel: CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  letterSpacing: "0.18em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
};

/** The title of a step inside a tool ("1. Pasta das imagens novas"). */
export const stepTitle: CSSProperties = {
  fontSize: 15,
  fontWeight: 600,
  color: "var(--vf-text)",
  lineHeight: 1.35,
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
  fontSize: 12,
  color: "var(--vf-text-dim)",
  lineHeight: 1.6,
};

/** Running text: explanations, help and notes. */
export const text: CSSProperties = {
  fontSize: 13,
  color: "var(--vf-text-dim)",
  lineHeight: 1.55,
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

export const danger: CSSProperties = { ...text, color: "var(--vf-danger, #f87171)" };

/** A warning that prevents a mistake: visible, never behind a disclosure. */
export const warn: CSSProperties = { ...text, color: "var(--vf-warn, #fbbf24)" };
