/** Breathing room kept between the end of a step and the top of the fixed bar. */
export const BAR_GAP = 16;

/**
 * Bottom padding the scroll area needs so the fixed bar never covers its end:
 * everything from the bar's top edge to the viewport bottom, plus a gap. This
 * follows the bar's real height, which grows when it wraps on narrow screens.
 */
export function barClearance(viewportHeight: number, barTop: number, gap = BAR_GAP): number {
  return Math.max(0, Math.ceil(viewportHeight - barTop + gap));
}
