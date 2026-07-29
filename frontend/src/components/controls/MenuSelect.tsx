import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import type { SelectOption } from "./SelectField";
import { useAnchoredMenu } from "./useAnchoredMenu";

interface MenuSelectProps {
  value: string;
  onChange: (v: string) => void;
  options: (string | SelectOption)[];
  /** Shown when `value` matches no option — also marks the action-menu use,
   * where the caller never stores the picked value ("+ adicionar filtro"). */
  placeholder?: string;
  /** Rendered accented when the current value is not the neutral one, so an
   * active filter is visible without reading the label. */
  activeWhenNot?: string;
  title?: string;
  minWidth?: number;
}

/** Compact dropdown for inline controls — the toolbar sibling of `SelectField`.
 *
 * Exists because a native `<select>` renders its popup with the operating
 * system's own widget: grey list, system font, OS highlight colour. Inside a
 * dark, monospaced UI that popup is the one surface that looks borrowed, and
 * `color-scheme: dark` only repaints its background, not its typography or
 * highlight. This draws the menu itself, so every list in the app matches.
 *
 * `SelectField` covers labelled form fields (full width, `FieldLabel`, room
 * for a `sub` line). This one covers toolbars: no label, sized to content.
 */
export function MenuSelect({
  value,
  onChange,
  options,
  placeholder,
  activeWhenNot,
  title,
  minWidth = 0,
}: MenuSelectProps) {
  const [open, setOpen] = useState(false);
  const { pos, wrapRef, btnRef, menuRef, closeRef } = useAnchoredMenu(open, minWidth);

  useEffect(() => {
    closeRef.current = () => setOpen(false);
  }, [closeRef]);

  // Escape closes — a portaled menu outlives its trigger's focus, so without
  // this it can only be dismissed by clicking away.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        setOpen(false);
      }
    };
    document.addEventListener("keydown", onKey, true);
    return () => document.removeEventListener("keydown", onKey, true);
  }, [open]);

  const normalize = (opt: string | SelectOption): SelectOption =>
    typeof opt === "string" ? { value: opt, label: opt } : opt;

  const current = options.map(normalize).find((o) => o.value === value);
  const label = current?.label ?? placeholder ?? value;
  const accented = activeWhenNot !== undefined && value !== activeWhenNot;

  return (
    <div ref={wrapRef} style={{ position: "relative", display: "inline-flex" }}>
      <button
        ref={btnRef}
        type="button"
        title={title}
        onClick={() => setOpen((o) => !o)}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 8,
          padding: "7px 10px",
          background: accented ? "var(--accent-soft)" : "rgba(0,0,0,0.35)",
          border: `1px solid ${
            open
              ? "var(--accent-vf)"
              : accented
                ? "var(--accent-vf)"
                : "var(--vf-panel-stroke)"
          }`,
          borderRadius: 8,
          color: "var(--vf-text)",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          cursor: "pointer",
          whiteSpace: "nowrap",
          transition: "border-color 160ms ease",
        }}
      >
        {label}
        <span
          style={{
            color: "var(--vf-text-muted)",
            transform: open ? "rotate(180deg)" : "none",
            transition: "transform 200ms ease",
            display: "inline-block",
            fontSize: 10,
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
              minWidth: pos.width,
              background: "rgba(14,16,22,0.96)",
              backdropFilter: "blur(12px)",
              WebkitBackdropFilter: "blur(12px)",
              border: "1px solid var(--vf-panel-stroke)",
              borderRadius: 10,
              padding: 5,
              // Above every overlay in the app (history/datasets sit at 100,
              // their confirmation layer at 200).
              zIndex: 1000,
              animation: "fadeUp 160ms ease",
              maxHeight: 280,
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
                    display: "block",
                    width: "100%",
                    padding: "8px 11px",
                    background: active ? "var(--accent-soft)" : "transparent",
                    border: "none",
                    borderRadius: 7,
                    textAlign: "left",
                    color: active ? "var(--vf-text)" : "var(--vf-text-dim)",
                    fontFamily: "var(--font-mono)",
                    fontSize: 12,
                    cursor: "pointer",
                    whiteSpace: "nowrap",
                    transition: "background 140ms ease",
                  }}
                  onMouseEnter={(e) => {
                    if (!active)
                      e.currentTarget.style.background = "rgba(255,255,255,0.05)";
                  }}
                  onMouseLeave={(e) => {
                    if (!active) e.currentTarget.style.background = "transparent";
                  }}
                >
                  {o.label}
                </button>
              );
            })}
          </div>,
          document.body,
        )}
    </div>
  );
}
