import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { FieldLabel } from "./FieldLabel";
import { useAnchoredMenu } from "./useAnchoredMenu";

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

/** Labelled dropdown with glass panel and accent highlight.
 *
 * Positioning and outside-click live in `useAnchoredMenu`, shared with the
 * compact `MenuSelect`. See that hook for why the menu is portaled.
 */
export function SelectField({
  label,
  value,
  onChange,
  options,
  hint,
}: SelectFieldProps) {
  const [open, setOpen] = useState(false);
  const { pos, wrapRef, btnRef, menuRef, closeRef } = useAnchoredMenu(open);

  useEffect(() => {
    closeRef.current = () => setOpen(false);
  }, [closeRef]);

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
