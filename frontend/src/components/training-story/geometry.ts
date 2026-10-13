/** The geometry of the training-story drawings (ADR-118): columns of neurons, the
 * curves that join them and the points arrows sit on. Pure, so the layouts are
 * covered by plain vitest and the components only draw what is computed here.
 *
 * Every drawing lives in the same box, `VIEW_W` by `VIEW_H` user units.
 */

export const VIEW_W = 640;
export const VIEW_H = 190;

export interface Pt {
  x: number;
  y: number;
}

/** A layer drawn as a column of neurons (circles) at one x. */
export interface Column {
  x: number;
  ys: number[];
  r: number;
}

const round = (n: number): number => Math.round(n * 10) / 10;

/** `n` neurons stacked around the middle of the box, `maxGap` apart at most. */
export function makeColumn(
  x: number,
  n: number,
  r = 6,
  maxGap = 30,
  top = 26,
  bottom = 164,
): Column {
  const mid = (top + bottom) / 2;
  if (n <= 1) return { x, ys: [mid], r };
  const gap = Math.min(maxGap, (bottom - top) / (n - 1));
  const span = gap * (n - 1);
  return {
    x,
    ys: Array.from({ length: n }, (_, i) => round(mid - span / 2 + i * gap)),
    r,
  };
}

/** The neuron of a column at a fraction (0 top, 1 bottom) of its height. */
export function neuronAt(col: Column, frac: number): Pt {
  const i = Math.round(Math.min(1, Math.max(0, frac)) * (col.ys.length - 1));
  return { x: col.x, y: col.ys[i] };
}

/** The S-curve between two points: it leaves and arrives horizontally. */
function controls(a: Pt, b: Pt): [Pt, Pt] {
  const xm = (a.x + b.x) / 2;
  return [
    { x: xm, y: a.y },
    { x: xm, y: b.y },
  ];
}

const f = (n: number): string => String(round(n));

/** `M a C ... b`: one connection. */
export function curve(a: Pt, b: Pt): string {
  const [c1, c2] = controls(a, b);
  return `M${f(a.x)} ${f(a.y)}C${f(c1.x)} ${f(c1.y)} ${f(c2.x)} ${f(c2.y)} ${f(b.x)} ${f(b.y)}`;
}

/** One path through several points, each pair joined by the same S-curve. */
export function chainPath(pts: readonly Pt[]): string {
  if (pts.length < 2) return "";
  let d = `M${f(pts[0].x)} ${f(pts[0].y)}`;
  for (let i = 1; i < pts.length; i += 1) {
    const [c1, c2] = controls(pts[i - 1], pts[i]);
    d += `C${f(c1.x)} ${f(c1.y)} ${f(c2.x)} ${f(c2.y)} ${f(pts[i].x)} ${f(pts[i].y)}`;
  }
  return d;
}

/** An arch from `a` to `b`, lifted `lift` above the higher of the two (or below,
 *  with a negative lift): skip connections, attention, the reconstruction. */
export function arcPath(a: Pt, b: Pt, lift: number): string {
  const top = lift >= 0 ? Math.min(a.y, b.y) - lift : Math.max(a.y, b.y) - lift;
  const xm = (a.x + b.x) / 2;
  return `M${f(a.x)} ${f(a.y)}Q${f(xm)} ${f(top)} ${f(b.x)} ${f(b.y)}`;
}

/** An arc between two neurons of the same column, bulging `off` to the right: the
 *  attention of one token to another. */
export function sideArc(x: number, y1: number, y2: number, off: number): string {
  return `M${f(x)} ${f(y1)}Q${f(x + off)} ${f((y1 + y2) / 2)} ${f(x)} ${f(y2)}`;
}

/** A point on the S-curve between `a` and `b`, `t` from 0 to 1. */
export function curvePoint(a: Pt, b: Pt, t: number): Pt {
  const [c1, c2] = controls(a, b);
  const u = 1 - t;
  const w = [u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t];
  return {
    x: w[0] * a.x + w[1] * c1.x + w[2] * c2.x + w[3] * b.x,
    y: w[0] * a.y + w[1] * c1.y + w[2] * c2.y + w[3] * b.y,
  };
}

/** The heading, in degrees, of the S-curve at `t` (0 is pointing right). */
export function curveAngle(a: Pt, b: Pt, t: number): number {
  const [c1, c2] = controls(a, b);
  const u = 1 - t;
  const dx =
    3 * u * u * (c1.x - a.x) + 6 * u * t * (c2.x - c1.x) + 3 * t * t * (b.x - c2.x);
  const dy =
    3 * u * u * (c1.y - a.y) + 6 * u * t * (c2.y - c1.y) + 3 * t * t * (b.y - c2.y);
  return (Math.atan2(dy, dx) * 180) / Math.PI;
}

/** Where a chevron sits on each segment of a chain: its middle, heading along it. */
export function chainMarks(pts: readonly Pt[]): Array<Pt & { angle: number }> {
  const marks: Array<Pt & { angle: number }> = [];
  for (let i = 1; i < pts.length; i += 1) {
    const p = curvePoint(pts[i - 1], pts[i], 0.5);
    marks.push({ x: round(p.x), y: round(p.y), angle: round(curveAngle(pts[i - 1], pts[i], 0.5)) });
  }
  return marks;
}

/** The pairs of neurons (by index) the faint web joins between two columns: each
 *  neuron to those of the next column at about its height, so the web reads as a
 *  tangle and not as a grid, and no neuron is left out. */
export function webLinks(a: Column, b: Column): Array<[number, number]> {
  const n = a.ys.length;
  const m = b.ys.length;
  const frac = (i: number, count: number): number => (count <= 1 ? 0.5 : i / (count - 1));
  const reach = 0.5 / Math.max(1, Math.max(n, m) - 1) + 0.22;
  const seen = new Set<string>();
  const pairs: Array<[number, number]> = [];
  const add = (i: number, j: number) => {
    const key = `${i},${j}`;
    if (seen.has(key)) return;
    seen.add(key);
    pairs.push([i, j]);
  };
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < m; j += 1) {
      if (Math.abs(frac(i, n) - frac(j, m)) <= reach) add(i, j);
    }
  }
  // A column that is much smaller than its neighbour would leave neurons out of
  // the threshold: each one still joins the neuron nearest its height.
  const nearest = (from: number, count: number, other: number): number => {
    let best = 0;
    for (let k = 1; k < other; k += 1) {
      if (Math.abs(frac(k, other) - frac(from, count)) < Math.abs(frac(best, other) - frac(from, count))) {
        best = k;
      }
    }
    return best;
  };
  for (let i = 0; i < n; i += 1) add(i, nearest(i, n, m));
  for (let j = 0; j < m; j += 1) add(nearest(j, m, n), j);
  return pairs;
}

/** One neuron of each column, at a fraction of its height: the way a signal goes
 *  through the network. `prefix` are the points it comes from before the first. */
export function chainThrough(cols: readonly Column[], frac: number, prefix: readonly Pt[] = []): Pt[] {
  return [...prefix, ...cols.map((c) => neuronAt(c, frac))];
}
