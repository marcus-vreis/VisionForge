import { useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

/** The "i" next to a field label, holding the explanation on demand.
 *
 * The explanations used to sit under every input as a paragraph. They are good
 * explanations — the problem is that a grid of fields with two-line captions
 * under some and one-line under others has rows of different heights, which is
 * what reads as "misaligned". Moving the text into a dot keeps the help one
 * pointer away and lets every field be the same height.
 *
 * **Portaled, for the same reason the dropdowns are** (see `useAnchoredMenu`):
 * the form cards use `backdrop-filter`, which creates a stacking context per
 * card, so an absolutely-positioned bubble is trapped inside its own card and
 * gets clipped at the edge. Portaling to <body> escapes that, at the cost of
 * positioning by hand — and the position is then clamped to the viewport, so a
 * field near either margin still shows its whole explanation.
 *
 * Opens on hover *and* on click: hover alone is unreachable on a touch screen,
 * and a tooltip nobody can open is decoration.
 */

const WIDTH = 260;
const MARGIN = 8;
const GAP = 8;

interface Placement {
  left: number;
  top: number;
  above: boolean;
}

export function InfoDot({ text }: { text: string }) {
  const [placement, setPlacement] = useState<Placement | null>(null);
  const btnRef = useRef<HTMLButtonElement>(null);
  const id = useId();
  const open = placement !== null;

  const show = () => {
    const el = btnRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    // Centre on the dot, then pull back inside whichever margin it crossed.
    const centred = r.left + r.width / 2 - WIDTH / 2;
    const left = Math.min(
      Math.max(MARGIN, centred),
      Math.max(MARGIN, window.innerWidth - WIDTH - MARGIN),
    );
    // Above by default; below when there is no room, which is the case for the
    // first field of a panel scrolled to the top.
    const above = r.top > 160;
    setPlacement({
      left,
      top: above ? r.top - GAP : r.bottom + GAP,
      above,
    });
  };

  const hide = () => setPlacement(null);

  // A tooltip glued to a element that moved is worse than no tooltip.
  useLayoutEffect(() => {
    if (!open) return;
    window.addEventListener("scroll", hide, true);
    window.addEventListener("resize", hide);
    return () => {
      window.removeEventListener("scroll", hide, true);
      window.removeEventListener("resize", hide);
    };
  }, [open]);

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        aria-label="Explicação"
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onClick={() => (open ? hide() : show())}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
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
      {placement &&
        createPortal(
          <span
            id={id}
            role="tooltip"
            style={{
              position: "fixed",
              left: placement.left,
              top: placement.top,
              transform: placement.above ? "translateY(-100%)" : "none",
              width: WIDTH,
              padding: "10px 12px",
              background: "oklch(0.18 0.01 260)",
              border: "1px solid var(--vf-panel-stroke)",
              borderRadius: 8,
              boxShadow: "0 8px 24px rgba(0,0,0,0.45)",
              color: "var(--vf-text-dim)",
              fontSize: 11,
              lineHeight: 1.5,
              letterSpacing: 0,
              textTransform: "none",
              zIndex: 9999,
              pointerEvents: "none",
              whiteSpace: "normal",
            }}
          >
            {text}
          </span>,
          document.body,
        )}
    </>
  );
}
