/**
 * Pure helpers of the shared line chart (components/ui/LineChart.tsx): ticks,
 * pointer snapping, tooltip placement and path building. No React, no text.
 *
 * Nothing here smooths or interpolates: a path joins the measured points with
 * straight segments and breaks wherever a value is missing.
 */

export interface DataPoint {
  x: number;
  /** Null, undefined or NaN: nothing was measured at this x. */
  y: number | null | undefined;
}

export interface Domain {
  min: number;
  max: number;
  /** Round values to label, ascending, inside [min, max]. */
  ticks: number[];
}

const isFiniteNumber = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** Smallest "round" step (1, 2, 2.5, 5 times a power of ten) that is >= raw. */
function niceStep(raw: number): number {
  const exp = Math.floor(Math.log10(raw));
  const base = 10 ** exp;
  const f = raw / base;
  const m = f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10;
  return m * base;
}

/** Strips float noise (0.30000000000000004) from a multiple of `step`. */
function snap(value: number, step: number): number {
  const decimals = Math.min(20, Math.max(0, -Math.floor(Math.log10(step)) + 2));
  return Number(value.toFixed(decimals));
}

/**
 * A y domain with 3 to 5 round ticks covering [min, max]. A series that never
 * moved (min === max) gets one tick at its value and a padded domain, so the
 * line runs through the middle with a single label. Non-finite input yields
 * null (nothing to draw).
 */
export function niceDomain(min: number, max: number, target = 4): Domain | null {
  if (!isFiniteNumber(min) || !isFiniteNumber(max)) return null;
  if (min > max) [min, max] = [max, min];
  if (min === max) {
    const pad = Math.abs(min) > 0 ? Math.abs(min) * 0.1 : 1;
    return { min: min - pad, max: max + pad, ticks: [min] };
  }
  // Tiny ranges relative to the values (1.0000001 vs 1.0000002) still resolve:
  // the step comes from the range, not from the magnitude of the values.
  let step = niceStep((max - min) / target);
  for (let guard = 0; guard < 8; guard += 1) {
    const lo = Math.floor(min / step + 1e-9) * step;
    const hi = Math.ceil(max / step - 1e-9) * step;
    const count = Math.round((hi - lo) / step) + 1;
    if (count <= 5) {
      const ticks = Array.from({ length: count }, (_, i) => snap(lo + i * step, step));
      return { min: ticks[0], max: ticks[ticks.length - 1], ticks };
    }
    step = niceStep(step * 1.01);
  }
  return { min, max, ticks: [min, max] };
}

/** The [min, max] of the finite values of every series, or null if there are none. */
export function valueExtent(
  series: ReadonlyArray<ReadonlyArray<DataPoint>>,
): { min: number; max: number } | null {
  let min = Infinity;
  let max = -Infinity;
  for (const points of series) {
    for (const p of points) {
      if (isFiniteNumber(p.y)) {
        if (p.y < min) min = p.y;
        if (p.y > max) max = p.y;
      }
    }
  }
  return min <= max ? { min, max } : null;
}

/** Sorted distinct x of every point that carries a value. */
export function uniqueXs(series: ReadonlyArray<ReadonlyArray<DataPoint>>): number[] {
  const seen = new Set<number>();
  for (const points of series) {
    for (const p of points) if (isFiniteNumber(p.x) && isFiniteNumber(p.y)) seen.add(p.x);
  }
  return [...seen].sort((a, b) => a - b);
}

/**
 * Integer x ticks: the first, the last and a few round ones between, at most
 * `maxCount` of them, never closer than half a step to the last.
 */
export function xTicks(xMin: number, xMax: number, maxCount = 6): number[] {
  if (!isFiniteNumber(xMin) || !isFiniteNumber(xMax)) return [];
  const lo = Math.round(Math.min(xMin, xMax));
  const hi = Math.round(Math.max(xMin, xMax));
  const span = hi - lo;
  if (span === 0) return [lo];
  const slots = Math.max(2, Math.min(maxCount, span + 1));
  const step = Math.max(1, Math.ceil(niceStep(span / (slots - 1))));
  const ticks: number[] = [];
  for (let v = lo; v < hi; v += step) ticks.push(v);
  // The last tick would crowd the end: the end takes its place.
  if (ticks.length > 1 && hi - ticks[ticks.length - 1] < step / 2) ticks.pop();
  ticks.push(hi);
  return ticks;
}

/**
 * Index of the position nearest to `px` in an ascending list (binary search).
 * Ties go to the lower index. -1 for an empty list or a non-finite pointer.
 */
export function nearestIndex(positions: ReadonlyArray<number>, px: number): number {
  if (positions.length === 0 || !isFiniteNumber(px)) return -1;
  let lo = 0;
  let hi = positions.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (positions[mid] <= px) lo = mid;
    else hi = mid;
  }
  if (positions.length === 1) return 0;
  return Math.abs(px - positions[lo]) <= Math.abs(positions[hi] - px) ? lo : hi;
}

/** Linear map of `value` from [d0, d1] to [r0, r1]; the middle when d0 === d1. */
export function scaleLinear(value: number, d0: number, d1: number, r0: number, r1: number): number {
  if (d0 === d1) return (r0 + r1) / 2;
  return r0 + ((value - d0) / (d1 - d0)) * (r1 - r0);
}

export interface TooltipPlacement {
  left: number;
  top: number;
  /** Which side of the guide line the card sits on. */
  side: "right" | "left";
}

/**
 * Where to put the tooltip card: beside the guide line, flipped to its left
 * when it would leave the chart on the right, and kept inside vertically.
 * `left` is the card's left edge.
 */
export function placeTooltip(opts: {
  anchorX: number;
  anchorY: number;
  width: number;
  height: number;
  chartWidth: number;
  chartHeight: number;
  gap?: number;
}): TooltipPlacement {
  const gap = opts.gap ?? 12;
  const rightEdge = opts.anchorX + gap + opts.width;
  const side: "right" | "left" = rightEdge <= opts.chartWidth ? "right" : "left";
  let left = side === "right" ? opts.anchorX + gap : opts.anchorX - gap - opts.width;
  left = Math.max(0, Math.min(left, Math.max(0, opts.chartWidth - opts.width)));
  let top = opts.anchorY - opts.height / 2;
  top = Math.max(0, Math.min(top, Math.max(0, opts.chartHeight - opts.height)));
  return { left, top, side };
}

const fmt = (n: number) => String(Math.round(n * 100) / 100);

/** Runs of consecutive points that carry a finite value (a gap splits a run). */
export function runsOf<T extends DataPoint>(points: ReadonlyArray<T>): T[][] {
  const runs: T[][] = [];
  let current: T[] = [];
  for (const p of points) {
    if (isFiniteNumber(p.x) && isFiniteNumber(p.y)) {
      current.push(p);
    } else if (current.length > 0) {
      runs.push(current);
      current = [];
    }
  }
  if (current.length > 0) runs.push(current);
  return runs;
}

/**
 * SVG path `d` joining the measured points (already in pixels) with straight
 * segments. A missing value starts a new sub-path; a run of one point has no
 * segment and is left to the caller to draw as a dot (see `isolatedPoints`).
 */
export function buildPath(points: ReadonlyArray<DataPoint>): string {
  return runsOf(points)
    .filter((run) => run.length > 1)
    .map((run) => run.map((p, i) => `${i === 0 ? "M" : "L"}${fmt(p.x)} ${fmt(p.y as number)}`).join(" "))
    .join(" ");
}

/** The points a line cannot show: measured, with a gap (or the edge) on both sides. */
export function isolatedPoints<T extends DataPoint>(points: ReadonlyArray<T>): T[] {
  return runsOf(points)
    .filter((run) => run.length === 1)
    .map((run) => run[0]);
}

/** The area under each run, closed down to `baseY`; for a single-series chart. */
export function buildAreaPath(points: ReadonlyArray<DataPoint>, baseY: number): string {
  return runsOf(points)
    .filter((run) => run.length > 1)
    .map((run) => {
      const line = run.map((p, i) => `${i === 0 ? "M" : "L"}${fmt(p.x)} ${fmt(p.y as number)}`).join(" ");
      const last = run[run.length - 1];
      return `${line} L${fmt(last.x)} ${fmt(baseY)} L${fmt(run[0].x)} ${fmt(baseY)} Z`;
    })
    .join(" ");
}
