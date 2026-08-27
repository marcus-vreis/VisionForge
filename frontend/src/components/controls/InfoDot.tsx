import { useId, useState } from "react";

/** The "i" next to a field label, holding the explanation on demand.
 *
 * The explanations used to sit under every input as a paragraph. They are good
 * explanations — the problem is that a grid of fields with two-line captions
 * under some and one-line under others has rows of different heights, which is
 * what reads as "misaligned". Moving the text into a dot keeps the help one
 * pointer away and lets every field be the same height.
 *
 * Opens on hover *and* on click: hover alone is unreachable on a touch screen,
 * and a tooltip nobody can open is decoration.
 */
export function InfoDot({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();

  return (
    <span style={{ position: "relative", display: "inline-flex" }}>
      <button
        type="button"
        aria-label="Explicação"
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        style={{
          width: 14,
          height: 14,
          borderRadius: "50%",
          border: "1px solid var(--vf-panel-stroke)",
          background: "rgba(255,255,255,0.04)",
          color: "var(--vf-text-muted)",
          fontFamily: "var(--font-mono)",
          fontSize: 9,
          lineHeight: 1,
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          cursor: "help",
          padding: 0,
          flexShrink: 0,
        }}
      >
        i
      </button>
      {open && (
        <span
          id={id}
          role="tooltip"
          style={{
            position: "absolute",
            bottom: "calc(100% + 8px)",
            left: "50%",
            transform: "translateX(-50%)",
            width: 260,
            padding: "10px 12px",
            background: "oklch(0.18 0.01 260)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 8,
            boxShadow: "0 8px 24px rgba(0,0,0,0.45)",
            color: "var(--vf-text-dim)",
            fontFamily: "var(--font-sans, inherit)",
            fontSize: 11,
            lineHeight: 1.5,
            letterSpacing: 0,
            textTransform: "none",
            zIndex: 50,
            pointerEvents: "none",
            whiteSpace: "normal",
          }}
        >
          {text}
        </span>
      )}
    </span>
  );
}
