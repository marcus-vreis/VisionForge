import type { ReactNode } from "react";

import { switchStyle } from "./trackerStyles";

/** One tracker's on/off: its brand mark, its name and a switch. */
export function TrackerSwitch({
  icon,
  label,
  checked,
  disabled,
  onChange,
}: {
  icon: ReactNode;
  label: string;
  checked: boolean;
  disabled: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        fontSize: 14,
        color: "var(--vf-text)",
        cursor: disabled ? "default" : "pointer",
      }}
    >
      <span
        aria-hidden="true"
        style={{
          display: "inline-flex",
          flexShrink: 0,
          color: checked ? "var(--accent-vf)" : "var(--vf-text-dim)",
        }}
      >
        {icon}
      </span>
      <span style={{ flex: 1, minWidth: 0 }}>{label}</span>
      <input
        type="checkbox"
        role="switch"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        style={switchStyle(checked, disabled)}
      />
    </label>
  );
}
