import { useEffect, useRef, useState } from "react";

import { useT } from "../i18n/useT";
import { GUIDES, type GuideId } from "../lib/guides";

/** The header's "guia" button, which opens a small menu of the guides (ADR-115).
 *
 * The menu closes on a pick, on Esc and on a click anywhere outside it. It is
 * built from the registry, so a guide added there appears here with no change.
 * `initialOpen` exists for the tests, which render it without a DOM to click.
 */
export function GuideMenu({
  onSelect,
  initialOpen = false,
}: {
  onSelect: (id: GuideId) => void;
  initialOpen?: boolean;
}) {
  const t = useT();
  const [open, setOpen] = useState(initialOpen);
  const root = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
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

  return (
    <div ref={root} style={{ position: "relative" }}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        title={t.header.guideTitle}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "9px 13px",
          background: "rgba(255,255,255,0.025)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 10,
          color: "var(--vf-text-dim)",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          cursor: "pointer",
          animation: "fadeUp 700ms ease both",
        }}
      >
        <span style={{ fontSize: 13, lineHeight: 1 }}>◎</span>
        {t.header.guide}
      </button>
      {open && (
        <div
          role="menu"
          aria-label={t.guides.menuLabel}
          style={{
            position: "absolute",
            top: "calc(100% + 8px)",
            right: 0,
            width: 280,
            padding: 6,
            background: "rgba(10,12,16,0.97)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 12,
            boxShadow: "0 24px 70px rgba(0,0,0,0.6)",
            zIndex: 10,
          }}
        >
          {GUIDES.map((guide) => (
            <button
              key={guide.id}
              type="button"
              role="menuitem"
              onClick={() => {
                setOpen(false);
                onSelect(guide.id);
              }}
              style={{
                display: "block",
                width: "100%",
                padding: "10px 12px",
                background: "transparent",
                border: "none",
                borderRadius: 8,
                textAlign: "left",
                cursor: "pointer",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.background = "rgba(255,255,255,0.05)")
              }
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
            >
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  letterSpacing: "0.12em",
                  textTransform: "uppercase",
                  color: "var(--vf-text)",
                }}
              >
                {guide.title(t)}
              </div>
              <div
                style={{
                  marginTop: 5,
                  fontSize: 12,
                  lineHeight: 1.5,
                  color: "var(--vf-text-muted)",
                }}
              >
                {guide.summary(t)}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
