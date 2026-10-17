/** Where everything sits in each family's drawing (ADR-118).
 *
 * A drawing is columns of neurons joined by a faint web of curves, plus what is
 * particular to the family: skip arcs (U-Net), attention arcs (transformer),
 * boxes at the heads (detector), a reconstruction arc (autoencoder), a memory
 * bank (PatchCore). The *chains* are the routes the particles take: forward
 * ones start at the data and go to the output, backward ones are written in the
 * forward order and walked in reverse by the renderer, so a gradient leaves from
 * the output and arrives at the input.
 *
 * Pure data and arithmetic: the components draw, the tests read the numbers.
 */

import type { StoryFamily, StoryOutput } from "../../lib/story-family";
import type { PhaseKey } from "../../lib/training-phase";
import type { StoryPhase, StoryStage } from "../../lib/training-story";
import {
  arcPath,
  chainThrough,
  makeColumn,
  neuronAt,
  sideArc,
  type Column,
  type Pt,
} from "./geometry";

export type ArcKind = "skip" | "attention" | "recon" | "lateral";

export interface Arc {
  d: string;
  kind: ArcKind;
}

/** A detector's output box. */
export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** A simple shape standing for an object in the detector's thumbnail. */
export type ThumbObject =
  | { shape: "circle"; cx: number; cy: number; r: number }
  | { shape: "ellipse"; cx: number; cy: number; rx: number; ry: number }
  | { shape: "rect"; x: number; y: number; w: number; h: number };

/** The small image the detector's heads end in: a frame and the objects in it. */
export interface Thumbnail {
  frame: Box;
  objects: ThumbObject[];
}

/** A drawing that trains by epochs: every family but PatchCore. */
export type EpochFamily = Exclude<StoryFamily, "patchcore">;

export interface Drawing {
  family: EpochFamily;
  columns: Column[];
  /** Pairs of columns (by index) whose neurons the faint web joins. */
  webs: Array<[number, number]>;
  arcs: Arc[];
  boxes: Box[];
  /** The detector's result image, where the boxes sit over the objects. */
  thumbnail: Thumbnail | null;
  /** Head to thumbnail links of the detector, one per head. */
  links: Array<[Pt, Pt]>;
  /** Where the data comes from, left of the first column. */
  dataSource: Pt;
  /** The short route of the data into the first column. */
  data: Pt[];
  /** Routes of the forward pass, from the data to the output. */
  forward: Pt[][];
  /** Routes of the gradient, written in forward order. */
  backward: Pt[][];
  /** The route of the validation pulse. */
  validation: Pt[];
  /** Routes that follow a skip arc: the arc's own path is walked. */
  skips: string[];
}

const DATA_SOURCE: Pt = { x: 22, y: 95 };

/** Adjacent column pairs `[0,1], [1,2], ...` up to `count` columns. */
function adjacent(count: number): Array<[number, number]> {
  return Array.from({ length: Math.max(0, count - 1) }, (_, i) => [i, i + 1] as [number, number]);
}

const FRACS_FORWARD = [0.2, 0.55, 0.9];
const FRACS_BACKWARD = [0.4, 0.8];

/** The routes every column-by-column drawing shares. */
function routes(columns: Column[]): Pick<Drawing, "forward" | "backward" | "validation" | "data"> {
  return {
    data: [DATA_SOURCE, neuronAt(columns[0], 0.5)],
    forward: FRACS_FORWARD.map((f) => chainThrough(columns, f, [DATA_SOURCE])),
    backward: FRACS_BACKWARD.map((f) => chainThrough(columns, f)),
    validation: chainThrough(columns, 0.35, [DATA_SOURCE]),
  };
}

function cnn(output: StoryOutput): Drawing {
  const columns = [
    makeColumn(90, 5),
    makeColumn(180, 6),
    makeColumn(270, 5),
    makeColumn(360, 4),
    makeColumn(450, 3),
    makeColumn(550, output === "value" ? 1 : 3, 7),
  ];
  return {
    family: "cnn",
    columns,
    webs: adjacent(columns.length),
    arcs: [],
    boxes: [],
    thumbnail: null,
    links: [],
    dataSource: DATA_SOURCE,
    skips: [],
    ...routes(columns),
  };
}

function transformer(output: StoryOutput): Drawing {
  const tokens = [90, 185, 280, 375].map((x) => makeColumn(x, 5, 5.5, 28));
  const columns = [
    ...tokens,
    // The class token the blocks hand the answer to, then the head.
    makeColumn(470, 1, 8),
    makeColumn(565, output === "value" ? 1 : 3, 7),
  ];
  // Attention: every token looks at others of its own block.
  const arcs: Arc[] = [];
  for (const col of tokens.slice(1)) {
    for (const [a, b, off] of [
      [0, 2, 16],
      [1, 3, 16],
      [2, 4, 16],
      [0, 4, 26],
    ] as const) {
      arcs.push({ d: sideArc(col.x + col.r * 0.4, col.ys[a], col.ys[b], off), kind: "attention" });
    }
  }
  return {
    family: "transformer",
    columns,
    webs: adjacent(columns.length),
    arcs,
    boxes: [],
    thumbnail: null,
    links: [],
    dataSource: DATA_SOURCE,
    skips: [],
    ...routes(columns),
  };
}

function encoderDecoder(): Drawing {
  const sizes = [6, 5, 4, 2, 4, 5, 6];
  const columns = [60, 140, 220, 320, 420, 500, 580].map((x, i) =>
    makeColumn(x, sizes[i], i === 3 ? 7 : 5.5),
  );
  const top = (i: number): Pt => neuronAt(columns[i], 0);
  const arcs: Arc[] = [
    { d: arcPath(top(2), top(4), 26), kind: "skip" },
    { d: arcPath(top(1), top(5), 24), kind: "skip" },
    { d: arcPath(top(0), top(6), 20), kind: "skip" },
  ];
  return {
    family: "encoder_decoder",
    columns,
    webs: adjacent(columns.length),
    arcs,
    boxes: [],
    thumbnail: null,
    links: [],
    dataSource: DATA_SOURCE,
    // The two outer skips carry the signal; the inner one is only drawn.
    skips: [arcs[1].d, arcs[2].d],
    ...routes(columns),
  };
}

function autoencoder(): Drawing {
  const sizes = [6, 5, 3, 2, 3, 5, 6];
  const columns = [60, 140, 220, 320, 420, 500, 580].map((x, i) =>
    makeColumn(x, sizes[i], i === 3 ? 7.5 : 5.5),
  );
  return {
    family: "autoencoder",
    columns,
    webs: adjacent(columns.length),
    // The output is compared with the input: that is the whole supervision.
    arcs: [{ d: arcPath(neuronAt(columns[6], 1), neuronAt(columns[0], 1), -20), kind: "recon" }],
    boxes: [],
    thumbnail: null,
    links: [],
    dataSource: DATA_SOURCE,
    skips: [],
    ...routes(columns),
  };
}

/** The result image of the detector: a frame with three objects of different sizes. */
const THUMBNAIL: Thumbnail = {
  frame: { x: 505, y: 38, w: 110, h: 114 },
  objects: [
    { shape: "circle", cx: 536, cy: 58, r: 6 },
    { shape: "rect", x: 572, y: 66, w: 30, h: 24 },
    { shape: "ellipse", cx: 548, cy: 119, rx: 28, ry: 17 },
  ],
};

/** The three boxes, small to large, each one over one object of the thumbnail. */
const DETECTOR_BOXES: Box[] = [
  { x: 527, y: 49, w: 18, h: 18 },
  { x: 567, y: 61, w: 40, h: 34 },
  { x: 516, y: 98, w: 64, h: 42 },
];

function detector(): Drawing {
  const backbone = [makeColumn(60, 5), makeColumn(135, 4), makeColumn(210, 3)];
  const neck = [makeColumn(290, 3), makeColumn(365, 3)];
  const heads = makeColumn(450, 3, 5.5, 50);
  const columns = [...backbone, ...neck, heads];
  const trunk = [...backbone, ...neck];
  const centre = (b: Box): Pt => ({ x: b.x + b.w / 2, y: b.y + b.h / 2 });
  // Each head enters the thumbnail's left edge at the height of its own box.
  const entry = (k: number): Pt => ({ x: THUMBNAIL.frame.x, y: centre(DETECTOR_BOXES[k]).y });
  const headRoute = (frac: number, k: number): Pt[] => [
    ...chainThrough(trunk, frac, [DATA_SOURCE]),
    neuronAt(heads, k / 2),
    entry(k),
    centre(DETECTOR_BOXES[k]),
  ];
  return {
    family: "detector",
    columns,
    webs: adjacent(columns.length),
    // The feature pyramid: a lateral path from the backbone into the neck.
    arcs: [{ d: arcPath(neuronAt(backbone[2], 1), neuronAt(neck[1], 1), -18), kind: "lateral" }],
    boxes: DETECTOR_BOXES,
    thumbnail: THUMBNAIL,
    links: [0, 1, 2].map((k): [Pt, Pt] => [neuronAt(heads, k / 2), entry(k)]),
    dataSource: DATA_SOURCE,
    data: [DATA_SOURCE, neuronAt(backbone[0], 0.5)],
    forward: [headRoute(0.15, 0), headRoute(0.5, 1), headRoute(0.85, 2)],
    backward: [
      [...chainThrough(trunk, 0.3), neuronAt(heads, 0)],
      [...chainThrough(trunk, 0.8), neuronAt(heads, 1)],
    ],
    // The third head: the first two carry the forward pass and the gradient's arrows.
    validation: headRoute(0.7, 2),
    skips: [],
  };
}

/** The drawing of a family that trains by epochs. */
export function drawingFor(family: EpochFamily, output: StoryOutput): Drawing {
  switch (family) {
    case "transformer":
      return transformer(output);
    case "encoder_decoder":
      return encoderDecoder();
    case "detector":
      return detector();
    case "autoencoder":
      return autoencoder();
    default:
      return cnn(output);
  }
}

// ── PatchCore ────────────────────────────────────────────────────────────────

export interface PatchCoreDrawing {
  extractor: Column[];
  /** The memory bank: dots in a grid. */
  bank: Pt[];
  bankRadius: number;
  /** The test image's patches, and the single score they end in. */
  query: Column;
  score: Pt;
  /** The three zones, in the order of the phases. */
  zones: Array<{ x: number; y: number; w: number; h: number }>;
  dataSource: Pt;
  data: Pt[];
  /** Routes per phase. */
  extract: Pt[][];
  store: Pt[][];
  scoring: Pt[][];
  /** Links from the query to its nearest dots of the bank. */
  nearest: Array<[Pt, Pt]>;
}

const BANK_COLS = 6;
const BANK_ROWS = 5;

export function patchcoreDrawing(): PatchCoreDrawing {
  const extractor = [makeColumn(62, 4, 5), makeColumn(122, 5, 5), makeColumn(182, 4, 5)];
  const bank: Pt[] = [];
  // Column-major, so a bank that fills up does it from left to right.
  for (let c = 0; c < BANK_COLS; c += 1) {
    for (let r = 0; r < BANK_ROWS; r += 1) {
      bank.push({ x: 292 + c * 26, y: 52 + r * 22 });
    }
  }
  const dot = (c: number, r: number): Pt => bank[c * BANK_ROWS + r];
  const query = makeColumn(505, 3, 5.5, 34);
  const score: Pt = { x: 590, y: 95 };
  const last = extractor[2];
  return {
    extractor,
    bank,
    bankRadius: 3.8,
    query,
    score,
    zones: [
      { x: 40, y: 22, w: 168, h: 146 },
      { x: 270, y: 22, w: 168, h: 146 },
      { x: 478, y: 22, w: 140, h: 146 },
    ],
    dataSource: DATA_SOURCE,
    data: [DATA_SOURCE, neuronAt(extractor[0], 0.5)],
    extract: [0.2, 0.55, 0.9].map((f) => chainThrough(extractor, f, [DATA_SOURCE])),
    store: [
      [neuronAt(last, 0.1), dot(0, 0)],
      [neuronAt(last, 0.5), dot(1, 2)],
      [neuronAt(last, 0.9), dot(0, 4)],
    ],
    scoring: [
      [dot(BANK_COLS - 1, 2), neuronAt(query, 0.5), score],
      [dot(BANK_COLS - 1, 0), neuronAt(query, 0), score],
    ],
    nearest: [
      [neuronAt(query, 0), dot(BANK_COLS - 1, 0)],
      [neuronAt(query, 0.5), dot(BANK_COLS - 1, 2)],
      [neuronAt(query, 1), dot(BANK_COLS - 1, 4)],
    ],
  };
}

/** The index of the zone a phase is in, or -1. */
export function zoneOfPhase(key: PhaseKey | null): number {
  switch (key) {
    case "extractingFeatures":
      return 0;
    case "buildingBank":
      return 1;
    case "scoring":
      return 2;
    default:
      return -1;
  }
}

/** How many dots of the memory bank are lit. Nothing is stored while the features
 *  are still being extracted; the bank fills with the progress of its own phase;
 *  scoring means it is full. A finished run shows a full bank (it was saved), a
 *  run that never got to the bank shows none. */
export function bankLit(stage: StoryStage, phase: StoryPhase | null, total = BANK_COLS * BANK_ROWS): number {
  if (stage === "done") return total;
  if (phase === null) return 0;
  switch (phase.key) {
    case "buildingBank":
      return Math.round(phase.fraction * total);
    case "scoring":
      return total;
    default:
      return 0;
  }
}
