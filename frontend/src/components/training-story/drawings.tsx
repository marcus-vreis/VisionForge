/** The drawings of the "Como funciona" tab, one per architecture family (ADR-118):
 * neon circuits on an organic topology. Layers are columns of neurons, joined by
 * a faint web of curves; the task's accent, amber and cyan are the three flows.
 *
 * The drawing never knows the time: it is given a stage, the flows that stage
 * carries and how they move (`StoryMotion`), and it draws exactly that. What
 * moves is illustrative. The only thing a drawing reads off the run is the number
 * of epochs reported, to fire one validation pulse per epoch.
 */

import { useMemo } from "react";
import type { StoryFamily, StoryOutput } from "../../lib/story-family";
import type { StoryFlows, StoryMotion, StoryPhase, StoryStage } from "../../lib/training-story";
import { FlowArrows, OneShotPulse, Pulse } from "./flow";
import {
  VIEW_H,
  VIEW_W,
  chainPath,
  curve,
  webLinks,
  type Column,
  type Pt,
} from "./geometry";
import {
  bankLit,
  drawingFor,
  patchcoreDrawing,
  zoneOfPhase,
  type Drawing,
  type EpochFamily,
} from "./layouts";
import type { Theme } from "../../lib/theme";
import { storyPalette, toneFor, type Tone } from "./tone";

/** What every drawing is given. */
export interface DrawingProps {
  accent: string;
  stage: StoryStage;
  flows: StoryFlows;
  motion: StoryMotion;
  /** Changes with every epoch reported: one validation pulse per change. */
  pulseKey: number;
  /** What a screen reader says the figure is. */
  label: string;
  /** The theme the page is painted in; the fixed colours follow it. Dark by default. */
  theme?: Theme;
}

function Frame({
  label,
  family,
  children,
}: {
  label: string;
  family: string;
  children: React.ReactNode;
}) {
  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      role="img"
      aria-label={label}
      data-family={family}
      preserveAspectRatio="xMidYMid meet"
      style={{ display: "block", width: "100%", height: "auto", maxHeight: 166 }}
    >
      {children}
    </svg>
  );
}

function Neurons({ columns, tone, dashed = false }: { columns: Column[]; tone: Tone; dashed?: boolean }) {
  return (
    <g
      fill={tone.line}
      fillOpacity={tone.fillOpacity}
      stroke={tone.line}
      strokeOpacity={tone.lineOpacity}
      strokeWidth={1.4}
      strokeDasharray={dashed ? "2.4 2.4" : undefined}
    >
      {columns.flatMap((col, c) =>
        col.ys.map((y, i) => <circle key={`${c}-${i}`} cx={col.x} cy={y} r={col.r} />),
      )}
    </g>
  );
}

/** The faint web: every neuron joined to those of the next column at about its
 *  height, by an S-curve. Computed once per drawing. */
function webPaths(columns: Column[], pairs: Array<[number, number]>): string {
  return pairs
    .flatMap(([a, b]) =>
      webLinks(columns[a], columns[b]).map(([i, j]) =>
        curve({ x: columns[a].x, y: columns[a].ys[i] }, { x: columns[b].x, y: columns[b].ys[j] }),
      ),
    )
    .join("");
}

/** Where the data comes from: three small stacked sheets. */
function DataSource({ at, tone }: { at: Pt; tone: Tone }) {
  return (
    <g fill={tone.line} fillOpacity={tone.fillOpacity + 0.04} stroke={tone.line} strokeOpacity={tone.lineOpacity} strokeWidth={1.2}>
      {[-5, 0, 5].map((dy, i) => (
        <rect key={i} x={at.x - 8 + i * 2} y={at.y - 7 + dy} width={14} height={9} rx={2} />
      ))}
    </g>
  );
}

/** The detector's result image: a frame with a few plain shapes as objects, and a
 *  link from each head. The boxes are drawn over it by the caller. */
function Thumbnail({
  thumbnail,
  links,
  tone,
}: {
  thumbnail: NonNullable<Drawing["thumbnail"]>;
  links: Array<[Pt, Pt]>;
  tone: Tone;
}) {
  const { frame, objects } = thumbnail;
  return (
    <g>
      {links.map(([from, to], i) => (
        <path
          key={i}
          data-part="head-link"
          data-head={i}
          d={curve(from, to)}
          fill="none"
          stroke={tone.line}
          strokeOpacity={tone.webOpacity * 1.9}
          strokeWidth={1.3}
        />
      ))}
      <rect
        data-part="thumbnail"
        x={frame.x}
        y={frame.y}
        width={frame.w}
        height={frame.h}
        rx={4}
        fill={tone.line}
        fillOpacity={tone.fillOpacity * 0.5}
        stroke={tone.line}
        strokeOpacity={tone.lineOpacity * 0.7}
        strokeWidth={1.2}
      />
      <g
        fill={tone.line}
        fillOpacity={tone.fillOpacity * 1.6}
        stroke={tone.line}
        strokeOpacity={tone.lineOpacity * 0.5}
        strokeWidth={1}
      >
        {objects.map((o, i) =>
          o.shape === "circle" ? (
            <circle key={i} data-part="thumb-object" cx={o.cx} cy={o.cy} r={o.r} />
          ) : o.shape === "ellipse" ? (
            <ellipse key={i} data-part="thumb-object" cx={o.cx} cy={o.cy} rx={o.rx} ry={o.ry} />
          ) : (
            <rect key={i} data-part="thumb-object" x={o.x} y={o.y} width={o.w} height={o.h} rx={3} />
          ),
        )}
      </g>
    </g>
  );
}

/** The pulses of a drawing that trains by epochs, or the arrows that stand in. */
function Flows({
  drawing,
  stage,
  flows,
  motion,
  pulseKey,
  accent,
  theme,
}: Pick<DrawingProps, "stage" | "flows" | "motion" | "pulseKey" | "accent" | "theme"> & { drawing: Drawing }) {
  if (motion === "none") return null;
  const palette = storyPalette(theme);
  const forward = drawing.forward.map(chainPath);
  const backward = drawing.backward.map((r) => chainPath([...r].reverse()));
  if (motion === "arrows") {
    return (
      <g>
        {flows.data && !flows.forward && <FlowArrows route={drawing.data} color={accent} />}
        {/* One route each, on three different rows so the arrows do not pile up. */}
        {flows.forward && <FlowArrows route={drawing.forward[0]} color={accent} />}
        {flows.backward && <FlowArrows route={drawing.backward[1]} color={palette.gradient} reverse />}
        {flows.validation && (
          <FlowArrows
            route={stage === "testing" ? drawing.validation : drawing.validation.slice(-2)}
            color={palette.validation}
          />
        )}
      </g>
    );
  }
  return (
    <g>
      {flows.data && !flows.forward && (
        <>
          <Pulse d={chainPath(drawing.data)} color={accent} dur={3.4} phase={0.3} />
          <Pulse d={chainPath(drawing.data)} color={accent} dur={3.4} phase={0.8} />
        </>
      )}
      {flows.forward &&
        forward.map((d, i) => (
          <Pulse key={`f${i}`} d={d} color={accent} dur={9} phase={[0.2, 0.55, 0.88][i % 3]} />
        ))}
      {flows.forward &&
        drawing.skips.map((d, i) => (
          <Pulse key={`s${i}`} d={d} color={accent} dur={7} phase={[0.4, 0.8][i % 2]} />
        ))}
      {flows.backward &&
        backward.map((d, i) => (
          <Pulse key={`b${i}`} d={d} color={palette.gradient} kind="ring" dur={11} phase={[0.35, 0.8][i % 2]} />
        ))}
      {flows.validation && stage === "testing" && (
        <>
          <Pulse d={chainPath(drawing.validation)} color={palette.validation} kind="diamond" dur={7} phase={0.3} />
          <Pulse d={chainPath(drawing.validation)} color={palette.validation} kind="diamond" dur={7} phase={0.8} />
        </>
      )}
      {flows.validation && stage === "training" && (
        <OneShotPulse d={chainPath(drawing.validation)} color={palette.validation} trigger={pulseKey} />
      )}
    </g>
  );
}

function EpochDrawing({
  drawing,
  ...props
}: DrawingProps & { drawing: Drawing }) {
  const { accent, stage, label, theme } = props;
  const tone = toneFor(stage, accent, theme);
  const web = useMemo(() => webPaths(drawing.columns, drawing.webs), [drawing]);
  return (
    <Frame label={label} family={drawing.family}>
      <path d={web} fill="none" stroke={tone.web} strokeOpacity={tone.webOpacity} strokeWidth={0.9} />
      {drawing.arcs.map((arc, i) => (
        <path
          key={i}
          d={arc.d}
          fill="none"
          stroke={tone.line}
          strokeOpacity={arc.kind === "attention" ? tone.webOpacity * 1.3 : tone.webOpacity * 1.9}
          strokeWidth={arc.kind === "attention" ? 0.9 : 1.3}
          strokeDasharray={arc.kind === "recon" ? "4 4" : undefined}
        />
      ))}
      {drawing.thumbnail && (
        <Thumbnail thumbnail={drawing.thumbnail} links={drawing.links} tone={tone} />
      )}
      {drawing.boxes.map((b, i) => (
        <rect
          key={i}
          data-part="box"
          data-head={i}
          x={b.x}
          y={b.y}
          width={b.w}
          height={b.h}
          rx={3}
          fill={tone.line}
          fillOpacity={tone.fillOpacity}
          stroke={tone.line}
          strokeOpacity={tone.lineOpacity}
          strokeWidth={1.6}
        />
      ))}
      <DataSource at={drawing.dataSource} tone={tone} />
      <Neurons columns={drawing.columns} tone={tone} />
      <Flows drawing={drawing} {...props} />
    </Frame>
  );
}

/** A drawing of a family that trains by epochs. */
function FamilyDrawing({
  name,
  output = "classes",
  ...props
}: DrawingProps & { name: EpochFamily; output?: StoryOutput }) {
  const drawing = useMemo(() => drawingFor(name, output), [name, output]);
  return <EpochDrawing drawing={drawing} {...props} />;
}

/** Columns narrowing towards a head: class scores, or one number for regression. */
export function CnnDrawing(props: DrawingProps & { output?: StoryOutput }) {
  return <FamilyDrawing name="cnn" {...props} />;
}
/** Patch tokens that attend to each other, ending in a class token. */
export function TransformerDrawing(props: DrawingProps & { output?: StoryOutput }) {
  return <FamilyDrawing name="transformer" {...props} />;
}
/** Shrinks to a bottleneck and grows back to a mask, with skip arcs. */
export function EncoderDecoderDrawing(props: DrawingProps) {
  return <FamilyDrawing name="encoder_decoder" {...props} />;
}
/** A backbone, a neck and three heads that end in boxes. */
export function DetectorDrawing(props: DrawingProps) {
  return <FamilyDrawing name="detector" {...props} />;
}
/** Squeezes through a bottleneck and rebuilds what it was given. */
export function AutoencoderDrawing(props: DrawingProps) {
  return <FamilyDrawing name="autoencoder" {...props} />;
}

/** The drawing of a family, chosen by name. */
export function StoryDrawing({
  family: name,
  output,
  phase,
  ...props
}: DrawingProps & { family: StoryFamily; output: StoryOutput; phase: StoryPhase | null }) {
  switch (name) {
    case "transformer":
      return <TransformerDrawing output={output} {...props} />;
    case "encoder_decoder":
      return <EncoderDecoderDrawing {...props} />;
    case "detector":
      return <DetectorDrawing {...props} />;
    case "autoencoder":
      return <AutoencoderDrawing {...props} />;
    case "patchcore":
      return <PatchCoreDrawing phase={phase} {...props} />;
    default:
      return <CnnDrawing output={output} {...props} />;
  }
}

/** PatchCore: no epochs and no gradient. It extracts features, stores them in a
 *  memory bank and scores an image by how far its patches are from the bank. */
export function PatchCoreDrawing({
  accent,
  stage,
  flows,
  motion,
  label,
  phase,
  theme,
}: DrawingProps & { phase: StoryPhase | null }) {
  const d = useMemo(() => patchcoreDrawing(), []);
  const tone = toneFor(stage, accent, theme);
  const palette = storyPalette(theme);
  const active = zoneOfPhase(phase?.key ?? null);
  const lit = bankLit(stage, phase);
  const running = stage === "training" || stage === "testing";
  const full = stage === "done" || stage === "testing";
  const scoring = full || active === 2;
  const route = active === 1 ? d.store : d.extract;
  const extractWeb = useMemo(() => webPaths(d.extractor, [[0, 1], [1, 2]]), [d]);
  const queryReach = d.query.ys.map((y) => curve({ x: d.query.x, y }, d.score)).join("");
  return (
    <Frame label={label} family="patchcore">
      {d.zones.map((z, i) => {
        const on = running ? active === i || (stage === "testing" && i === 2) : full;
        return (
          <rect
            key={i}
            x={z.x}
            y={z.y}
            width={z.w}
            height={z.h}
            rx={12}
            fill={tone.line}
            fillOpacity={on ? 0.05 : 0}
            stroke={tone.line}
            strokeOpacity={on ? 0.5 : 0.14}
            strokeWidth={1}
            strokeDasharray="3 5"
          />
        );
      })}
      <path d={extractWeb} fill="none" stroke={tone.web} strokeOpacity={tone.webOpacity} strokeWidth={0.9} />
      <DataSource at={d.dataSource} tone={tone} />
      <Neurons columns={d.extractor} tone={tone} dashed />
      <g fill={accent} stroke={accent} strokeWidth={1.2}>
        {d.bank.map((p, i) => (
          <circle
            key={i}
            cx={p.x}
            cy={p.y}
            r={d.bankRadius}
            fillOpacity={i < lit ? 0.85 : 0.05}
            strokeOpacity={i < lit ? 0.9 : tone.lineOpacity * 0.6}
            stroke={stage === "interrupted" || stage === "failed" ? tone.line : accent}
          />
        ))}
      </g>
      {scoring && (
        <g fill="none" stroke={palette.validation} strokeOpacity={0.28} strokeWidth={1}>
          {d.nearest.map(([a, b], i) => (
            <path key={i} d={curve(b, a)} />
          ))}
        </g>
      )}
      <path d={queryReach} fill="none" stroke={tone.web} strokeOpacity={tone.webOpacity} strokeWidth={0.9} />
      <Neurons columns={[d.query]} tone={scoring ? toneFor("training", palette.validation, theme) : tone} />
      <circle
        cx={d.score.x}
        cy={d.score.y}
        r={9}
        fill={palette.validation}
        fillOpacity={scoring ? 0.35 : 0.04}
        stroke={scoring ? palette.validation : palette.idle}
        strokeOpacity={scoring ? 0.9 : 0.3}
        strokeWidth={1.6}
      />
      {motion === "particles" && (
        <g>
          {flows.data && !flows.forward && (
            <Pulse d={chainPath(d.data)} color={accent} dur={3.4} phase={0.5} />
          )}
          {flows.forward &&
            route.map((r, i) => (
              <Pulse key={`${active}-${i}`} d={chainPath(r)} color={accent} dur={active === 1 ? 5 : 8} phase={[0.2, 0.55, 0.88][i % 3]} />
            ))}
          {flows.validation &&
            d.scoring.map((r, i) => (
              <Pulse key={`v${i}`} d={chainPath(r)} color={palette.validation} kind="diamond" dur={6} phase={[0.3, 0.8][i % 2]} />
            ))}
        </g>
      )}
      {motion === "arrows" && (
        <g>
          {flows.data && !flows.forward && <FlowArrows route={d.data} color={accent} />}
          {flows.forward && <FlowArrows route={route[1] ?? route[0]} color={accent} />}
          {flows.validation && <FlowArrows route={d.scoring[0]} color={palette.validation} />}
        </g>
      )}
    </Frame>
  );
}
