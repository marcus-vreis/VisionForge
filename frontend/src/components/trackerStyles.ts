import type { CSSProperties } from "react";

/** Shared look of the profile's integration settings (13px floor in the menu). */
export const TRACKER_FIELD: CSSProperties = {
  background: "var(--vf-panel)",
  color: "var(--vf-text)",
  padding: "6px 8px",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 6,
  fontSize: 13,
  width: "100%",
  boxSizing: "border-box",
};

export const TRACKER_LABEL: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 4,
};

export const TRACKER_HINT: CSSProperties = {
  fontSize: 13,
  lineHeight: 1.5,
  color: "var(--vf-text-muted)",
};

/** The track and knob drawn on the native checkbox itself, so the input keeps
 *  its checked state, focus ring and label while reading as an on/off switch. */
export function switchStyle(checked: boolean, disabled = false): CSSProperties {
  const knob = checked ? "calc(100% - 9px)" : "9px";
  return {
    appearance: "none",
    WebkitAppearance: "none",
    margin: 0,
    width: 32,
    height: 18,
    flexShrink: 0,
    borderRadius: 999,
    background: `radial-gradient(circle at ${knob} 50%, #fff 6px, transparent 6.5px), ${
      checked ? "var(--accent-vf)" : "rgb(var(--lt-ink, 255 255 255) / 0.16)"
    }`,
    boxShadow: checked ? "0 0 10px var(--accent-glow)" : "none",
    opacity: disabled ? 0.6 : 1,
    cursor: disabled ? "default" : "pointer",
    transition: "background 200ms ease, box-shadow 200ms ease",
  };
}
