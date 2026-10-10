/** Where a floating guide card sits (ADR-115). Pure geometry, like `placeCard`.
 *
 * A floating step is about watching a screen the card must not hide, so the
 * card goes beside its target when a side has room for it, and otherwise in the
 * bottom-left corner. The corner is the left one because the training sheet
 * keeps its buttons (stop, minimise, view results) at the bottom right, and the
 * last of them is what the researcher has to click at the end of the run.
 */

import { type Placement, type Viewport } from "../tour";

/** Narrower than the tour's card, so it clears a 720px sheet on a 1440px window. */
export const FLOATING_WIDTH = 340;
const MARGIN = 20;
const GAP = 16;

export function placeFloating(
  rect: DOMRect | null,
  height: number,
  view: Viewport,
): Placement {
  const corner: Placement = {
    left: MARGIN,
    top: Math.max(MARGIN, view.height - height - MARGIN),
  };
  if (!rect) return corner;

  const top = Math.min(
    Math.max(MARGIN, rect.top + rect.height / 2 - height / 2),
    Math.max(MARGIN, view.height - height - MARGIN),
  );
  const room = FLOATING_WIDTH + GAP + MARGIN;
  if (view.width - rect.right >= room) {
    return { left: rect.right + GAP, top };
  }
  if (rect.left >= room) {
    return { left: rect.left - GAP - FLOATING_WIDTH, top };
  }
  return corner;
}
