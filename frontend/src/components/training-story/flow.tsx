/** The moving parts of the drawings (ADR-118): luminous pulses that travel along a
 * curve, and the still arrows that stand in for them when the reader asked for
 * less motion.
 *
 * The pulses use SMIL `animateMotion`: no script runs per frame and nothing is
 * measured, so the cost is the browser's own and stops when the elements are
 * unmounted (which is how the drawings pause). Each flow has its own shape as well
 * as its own colour, so the three can be told apart without telling the colours
 * apart: a filled dot is data and the forward pass, a ring is the gradient, a
 * diamond is validation.
 */

import { useEffect, useRef } from "react";
import { chainMarks, type Pt } from "./geometry";

export type Glyph = "dot" | "ring" | "diamond";

/** How far behind the head the two trail dots run, in seconds of the loop. */
const TRAIL_LAG = [0.28, 0.56];

function GlyphShape({ kind, color }: { kind: Glyph; color: string }) {
  if (kind === "ring") {
    return (
      <>
        <circle r={9} fill="none" stroke={color} strokeOpacity={0.18} strokeWidth={3.5} />
        <circle r={5} fill="none" stroke={color} strokeWidth={1.9} />
      </>
    );
  }
  if (kind === "diamond") {
    return (
      <>
        <rect x={-7.5} y={-7.5} width={15} height={15} transform="rotate(45)" fill={color} fillOpacity={0.18} />
        <rect x={-4} y={-4} width={8} height={8} transform="rotate(45)" fill={color} />
      </>
    );
  }
  return (
    <>
      <circle r={9.5} fill={color} fillOpacity={0.14} />
      <circle r={5.6} fill={color} fillOpacity={0.3} />
      <circle r={3} fill={color} />
    </>
  );
}

/** The swatch of the legend: the same shape and colour the flow has in the drawing. */
export function LegendGlyph({ kind, color }: { kind: Glyph; color: string }) {
  return (
    <svg width={20} height={20} viewBox="-10.5 -10.5 21 21" aria-hidden="true" style={{ flexShrink: 0 }}>
      <GlyphShape kind={kind} color={color} />
    </svg>
  );
}

const FADE = { values: "0;1;1;0", keyTimes: "0;0.1;0.88;1" };

/** Seconds after the head at which each part of a one-shot pulse starts. */
const ONE_SHOT_LAGS = [0, TRAIL_LAG[0] * 2];

/** One pulse that loops along `d`, slowly. `phase` (0..1) is where in the loop it
 *  starts, so several pulses on a route are spread out and none begins at once. */
export function Pulse({
  d,
  color,
  kind = "dot",
  dur,
  phase,
}: {
  d: string;
  color: string;
  kind?: Glyph;
  dur: number;
  phase: number;
}) {
  // A negative begin starts the loop already under way. The trail begins later
  // than the head, so the head must be at least as far along as the longest lag.
  const lead = Math.max(phase * dur, TRAIL_LAG[1] + 0.1);
  const items = [
    { lag: 0, glyph: <GlyphShape kind={kind} color={color} /> },
    { lag: TRAIL_LAG[0], glyph: <circle r={2.3} fill={color} fillOpacity={0.5} /> },
    { lag: TRAIL_LAG[1], glyph: <circle r={1.6} fill={color} fillOpacity={0.28} /> },
  ];
  return (
    <>
      {items.map(({ lag, glyph }, i) => {
        const begin = `${-(lead - lag)}s`;
        return (
          <g key={i}>
            <animateMotion path={d} dur={`${dur}s`} begin={begin} repeatCount="indefinite" />
            <g opacity={0}>
              <animate
                attributeName="opacity"
                values={FADE.values}
                keyTimes={FADE.keyTimes}
                dur={`${dur}s`}
                begin={begin}
                repeatCount="indefinite"
              />
              {glyph}
            </g>
          </g>
        );
      })}
    </>
  );
}

/** The validation pulse of one epoch: it crosses the network once, when
 *  `trigger` changes (an `epoch_end`), and is gone until the next. The first
 *  value seen is not a change, so opening the tab on a run in progress does not
 *  fire one for an epoch that ended minutes ago. */
export function OneShotPulse({
  d,
  color,
  trigger,
  dur = 3.6,
}: {
  d: string;
  color: string;
  trigger: number;
  dur?: number;
}) {
  const shown = useRef(trigger);
  const motion = useRef<Array<SVGAnimationElement | null>>([]);
  const fade = useRef<Array<SVGAnimationElement | null>>([]);
  const glyphs = [
    <GlyphShape key="head" kind="diamond" color={color} />,
    <circle key="trail" r={2.3} fill={color} fillOpacity={0.45} />,
  ];
  useEffect(() => {
    if (shown.current === trigger) return;
    shown.current = trigger;
    // The trail starts a beat after the head.
    ONE_SHOT_LAGS.forEach((lag, i) => {
      motion.current[i]?.beginElementAt(lag);
      fade.current[i]?.beginElementAt(lag);
    });
  }, [trigger]);
  return (
    <>
      {glyphs.map((glyph, i) => (
        <g key={i}>
          <animateMotion
            ref={(el) => {
              motion.current[i] = el as SVGAnimationElement | null;
            }}
            path={d}
            dur={`${dur}s`}
            begin="indefinite"
            fill="remove"
          />
          <g opacity={0}>
            <animate
              ref={(el) => {
                fade.current[i] = el as SVGAnimationElement | null;
              }}
              attributeName="opacity"
              values={FADE.values}
              keyTimes={FADE.keyTimes}
              dur={`${dur}s`}
              begin="indefinite"
              fill="remove"
            />
            {glyph}
          </g>
        </g>
      ))}
    </>
  );
}

/** A chevron on each segment of a route, pointing the way the flow goes: the
 *  still drawing's answer to a pulse. `reverse` walks the route backwards, for the
 *  gradient. */
export function FlowArrows({
  route,
  color,
  reverse = false,
}: {
  route: readonly Pt[];
  color: string;
  reverse?: boolean;
}) {
  const pts = reverse ? [...route].reverse() : route;
  return (
    <g fill="none" stroke={color} strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round">
      {chainMarks(pts).map((m, i) => (
        <path
          key={i}
          d="M-4.2 -3.6L1.8 0L-4.2 3.6"
          transform={`translate(${m.x} ${m.y}) rotate(${m.angle})`}
        />
      ))}
    </g>
  );
}
