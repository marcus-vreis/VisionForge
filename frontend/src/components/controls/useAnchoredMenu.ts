import { useEffect, useLayoutEffect, useRef, useState } from "react";

export interface AnchoredPosition {
  left: number;
  top: number;
  width: number;
}

/** Anchor a portaled dropdown menu to its trigger button.
 *
 * The menu has to live in a portal: the form cards use `backdrop-filter`,
 * which creates a stacking context per card, so an absolutely-positioned menu
 * would be trapped inside its own card and painted under the following one.
 * Portaling to <body> escapes every card stacking context, at the cost of
 * having to position the menu by hand and keep it glued to the trigger.
 *
 * The scroll listener is registered in the capture phase so it fires for any
 * scrolling ancestor, not only the window — overlays scroll their own body.
 */
export function useAnchoredMenu(open: boolean, minWidth = 0) {
  const [pos, setPos] = useState<AnchoredPosition | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const btnRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<(() => void) | null>(null);

  useLayoutEffect(() => {
    if (!open) return;
    const reposition = () => {
      const el = btnRef.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      setPos({ left: r.left, top: r.bottom + 6, width: Math.max(r.width, minWidth) });
    };
    reposition();
    window.addEventListener("scroll", reposition, true);
    window.addEventListener("resize", reposition);
    return () => {
      window.removeEventListener("scroll", reposition, true);
      window.removeEventListener("resize", reposition);
    };
  }, [open, minWidth]);

  // Outside click closes. The menu is portaled out of the wrapper, so both the
  // trigger and the menu itself have to be excluded.
  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      const t = e.target as Node;
      if (wrapRef.current?.contains(t)) return;
      if (menuRef.current?.contains(t)) return;
      closeRef.current?.();
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  return { pos, wrapRef, btnRef, menuRef, closeRef };
}
