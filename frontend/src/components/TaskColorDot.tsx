import { useEffect, useRef, useState } from "react";

import { useT } from "../i18n/useT";
import { TASK_COLOR_PRESETS, isHexColor } from "../lib/task-color";

interface TaskColorDotProps {
  /** The task's current color. */
  color: string;
  /** The color the task registered with, which "reset" returns to. */
  registered: string;
  taskLabel: string;
  onChange: (color: string | null) => void;
}

/** A dot in the header that picks the color of the open researcher task (ADR-137). */
export function TaskColorDot({ color, registered, taskLabel, onChange }: TaskColorDotProps) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);

  // Closes on a click outside and on Escape, like the guide menu.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!boxRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const pick = (next: string | null) => {
    onChange(next);
    setOpen(false);
  };

  return (
    <div ref={boxRef} style={{ position: "relative", animation: "fadeUp 700ms ease both" }}>
      <button
        type="button"
        aria-label={t.taskColor.label(taskLabel)}
        aria-haspopup="dialog"
        aria-expanded={open}
        title={t.taskColor.label(taskLabel)}
        onClick={() => setOpen((v) => !v)}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          width: 36,
          height: 36,
          background: "rgb(var(--lt-ink, 255 255 255) / 0.025)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 10,
          cursor: "pointer",
        }}
      >
        <span
          style={{
            width: 14,
            height: 14,
            borderRadius: "50%",
            background: color,
            boxShadow: `0 0 0 2px var(--vf-bg), 0 0 10px ${color}`,
          }}
        />
      </button>

      {open && (
        <div
          role="dialog"
          aria-label={t.taskColor.label(taskLabel)}
          style={{
            position: "absolute",
            top: 44,
            right: 0,
            zIndex: 30,
            width: 212,
            padding: 12,
            display: "flex",
            flexDirection: "column",
            gap: 10,
            background: "var(--popover)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 12,
            boxShadow: "0 12px 32px rgb(var(--lt-shade, 0 0 0) / 0.35)",
          }}
        >
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              color: "var(--vf-text-muted)",
            }}
          >
            {t.taskColor.title}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8 }}>
            {TASK_COLOR_PRESETS.map((swatch) => {
              const on = swatch.toLowerCase() === color.toLowerCase();
              return (
                <button
                  key={swatch}
                  type="button"
                  aria-label={swatch}
                  aria-pressed={on}
                  onClick={() => pick(swatch)}
                  style={{
                    width: 28,
                    height: 28,
                    borderRadius: "50%",
                    background: swatch,
                    border: "none",
                    cursor: "pointer",
                    boxShadow: on ? `0 0 0 2px var(--popover), 0 0 0 4px ${swatch}` : "none",
                  }}
                />
              );
            })}
          </div>
          <label
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 8,
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              color: "var(--vf-text-dim)",
            }}
          >
            {t.taskColor.other}
            <input
              type="color"
              value={isHexColor(color) ? color : registered}
              onChange={(e) => onChange(e.target.value)}
              style={{ width: 40, height: 26, padding: 0, border: "none", background: "none", cursor: "pointer" }}
            />
          </label>
          <button
            type="button"
            onClick={() => pick(null)}
            disabled={color.toLowerCase() === registered.toLowerCase()}
            style={{
              padding: "6px 8px",
              background: "none",
              border: "1px solid var(--vf-panel-stroke)",
              borderRadius: 8,
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              color: "var(--vf-text-dim)",
              cursor: "pointer",
            }}
          >
            {t.taskColor.reset}
          </button>
        </div>
      )}
    </div>
  );
}
