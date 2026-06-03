import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { FieldLabel } from "./FieldLabel";

export interface SelectOption {
  value: string;
  label: string;
  sub?: string;
}

interface SelectFieldProps {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: (string | SelectOption)[];
  hint?: string;
}

/** Custom dropdown with glass panel and accent highlight.
 *
 * The open menu is rendered in a portal with fixed positioning. The form cards
 * use `backdrop-filter`, which creates a stacking context per card — an
 * absolutely-positioned menu would be trapped inside its own card and painted
 * *under* the following card (so the model list opened behind the Dataset
 * panel). Portaling to <body> escapes every card stacking context. */
export function SelectField({
  label,
  value,
  onChange,
  options,
  hint,
}: SelectFieldProps) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{
    left: number;
    top: number;
    width: number;
  } | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const btnRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  const reposition = () => {
    const el = btnRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    setPos({ left: r.left, top: r.bottom + 6, width: r.width });
  };

  // Position the portal menu against the button's viewport rect, and keep it
  // glued there while scrolling/resizing (capture phase catches scroll on any
  // ancestor container, not just window).
  useLayoutEffect(() => {
    if (!open) return;
    reposition();
    const onMove = () => reposition();
    window.addEventListener("scroll", onMove, true);
    window.addEventListener("resize", onMove);
    return () => {
      window.removeEventListener("scroll", onMove, true);
      window.removeEventListener("resize", onMove);
    };
  }, [open]);

  // Close on outside click — the menu lives outside the wrapper now, so check
  // both the trigger wrapper and the portaled menu before closing.
  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      const t = e.target as Node;
      if (wrapRef.current?.contains(t)) return;
      if (menuRef.current?.contains(t)) return;
      setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const normalize = (opt: string | SelectOption): SelectOption =>
    typeof opt === "string" ? { value: opt, label: opt } : opt;

  const current = options.map(normalize).find((o) => o.value === value);
  const currentLabel = current?.label ?? value;

  return (
    <div ref={wrapRef} style={{ position: "relative" }}>
      <FieldLabel dot hint={hint}>
        {label}
      </FieldLabel>
      <button
        ref={btnRef}
        type="button"
        onClick={() => setOpen((o) => !o)}
        style={{
          display: "flex",
          alignItems: "center",
          width: "100%",
          background: "rgba(12,14,18,0.65)",
          border: open
            ? "1px solid var(--accent-vf)"
            : "1px solid var(--vf-panel-stroke)",
          boxShadow: open ? "0 0 0 4px var(--accent-soft)" : "none",
          borderRadius: 10,
          padding: 0,
          transition: "border-color 160ms ease, box-shadow 160ms ease",
          cursor: "pointer",
        }}
      >
        <span
          style={{
            padding: "12px 14px",
            flex: 1,
            textAlign: "left",
            fontFamily: "var(--font-mono)",
            fontSize: 13,
            color: "var(--vf-text)",
            letterSpacing: "0.01em",
          }}
        >
          {currentLabel}
        </span>
        <span
          style={{
            paddingRight: 14,
            color: "var(--vf-text-muted)",
            transform: open ? "rotate(180deg)" : "none",
            transition: "transform 200ms ease",
            display: "inline-block",
          }}
        >
          ▾
        </span>
      </button>

      {open &&
        pos &&
        createPortal(
          <div
            ref={menuRef}
            style={{
              position: "fixed",
              top: pos.top,
              left: pos.left,
              width: pos.width,
              background: "rgba(14,16,22,0.96)",
              backdropFilter: "blur(12px)",
              border: "1px solid var(--vf-panel-stroke)",
              borderRadius: 10,
              padding: 6,
              zIndex: 1000,
              animation: "fadeUp 160ms ease",
              maxHeight: 260,
              overflowY: "auto",
              boxShadow: "0 24px 60px rgba(0,0,0,0.55)",
            }}
          >
            {options.map((opt) => {
              const o = normalize(opt);
              const active = o.value === value;
              return (
                <button
                  key={o.value}
                  type="button"
                  onClick={() => {
                    onChange(o.value);
                    setOpen(false);
                  }}
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "flex-start",
                    gap: 2,
                    width: "100%",
                    padding: "10px 12px",
                    background: active ? "var(--accent-soft)" : "transparent",
                    border: "none",
                    borderRadius: 7,
                    textAlign: "left",
                    color: "var(--vf-text)",
                    fontFamily: "var(--font-mono)",
                    fontSize: 13,
                    cursor: "pointer",
                    transition: "background 140ms ease",
                  }}
                  onMouseEnter={(e) => {
                    if (!active)
                      (e.currentTarget as HTMLButtonElement).style.background =
                        "rgba(255,255,255,0.04)";
                  }}
                  onMouseLeave={(e) => {
                    if (!active)
                      (e.currentTarget as HTMLButtonElement).style.background =
                        "transparent";
                  }}
                >
                  <span>{o.label}</span>
                  {o.sub && (
                    <span
                      style={{
                        fontSize: 10.5,
                        color: "var(--vf-text-muted)",
                        letterSpacing: 0,
                      }}
                    >
                      {o.sub}
                    </span>
                  )}
                </button>
              );
            })}
          </div>,
          document.body,
        )}
    </div>
  );
}
