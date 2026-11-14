/**
 * Line chart of per-epoch series, pure SVG. Used by the live "Resultados" tab and
 * the History comparison.
 *
 * Lines join the measured points as they are (no smoothing, no interpolation) and
 * break where a series has no value. Hovering, touching or arrowing over the plot
 * snaps to the nearest epoch: a guide line, a dot per series and a tooltip with
 * the exact values. There are no transitions, so `prefers-reduced-motion` has
 * nothing to switch off.
 */

import { useEffect, useId, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";

import { useT } from "../../i18n/useT";
import {
  buildAreaPath,
  buildPath,
  isolatedPoints,
  nearestIndex,
  niceDomain,
  placeTooltip,
  scaleLinear,
  uniqueXs,
  valueExtent,
  xTicks,
  type DataPoint,
} from "../../lib/line-chart";

export interface LineChartSeries {
  key: string;
  name: string;
  color: string;
  points: ReadonlyArray<DataPoint>;
}

export interface LineChartMarker {
  x: number;
  color: string;
  label?: string;
}

interface LineChartProps {
  /** Accessible name of the chart. */
  title: string;
  series: ReadonlyArray<LineChartSeries>;
  /** Formats a value for ticks and tooltip. */
  formatValue: (value: number) => string;
  /** Pixel height of the SVG; the width follows the container. */
  height?: number;
  /** Width used until the container has been measured (and in static markup). */
  width?: number;
  /** A dashed vertical line, e.g. the best epoch. */
  marker?: LineChartMarker | null;
}

const FONT = 11;
const MONO = "var(--font-mono)";
const visuallyHidden = {
  position: "absolute",
  width: 1,
  height: 1,
  margin: -1,
  padding: 0,
  overflow: "hidden",
  clip: "rect(0 0 0 0)",
  whiteSpace: "nowrap",
  border: 0,
} as const;

export function LineChart({
  title,
  series: all,
  formatValue,
  height = 150,
  width: initialWidth = 640,
  marker = null,
}: LineChartProps) {
  const c = useT().chart;
  const hintId = useId();
  const box = useRef<HTMLDivElement>(null);
  const [measured, setMeasured] = useState<number | null>(null);
  const [active, setActive] = useState<number | null>(null);

  // Real pixels, so the 11px text stays 11px whatever the container's width.
  useEffect(() => {
    const el = box.current;
    if (!el) return undefined;
    const read = () => {
      // A hidden tab measures 0: keep the last width then.
      if (el.clientWidth > 0) setMeasured(Math.max(240, Math.round(el.clientWidth)));
    };
    read();
    if (typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(read);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const series = all.filter((s) => s.points.some((p) => Number.isFinite(p.y as number)));
  const extent = valueExtent(series.map((s) => s.points));
  const domain = extent ? niceDomain(extent.min, extent.max) : null;
  const xs = uniqueXs(series.map((s) => s.points));
  if (!extent || !domain || xs.length === 0) return null;

  const width = measured ?? initialWidth;
  const tickLabels = domain.ticks.map((t) => formatValue(t));
  const labelChars = Math.max(...tickLabels.map((l) => l.length));
  const pad = {
    top: 12,
    right: 14,
    bottom: 26,
    left: Math.max(44, Math.ceil(labelChars * FONT * 0.62) + 14),
  };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const xMin = xs[0];
  const xMax = xs[xs.length - 1];
  // One epoch only: it sits in the middle instead of on the axis.
  const px = (x: number) =>
    xMin === xMax ? pad.left + innerW / 2 : scaleLinear(x, xMin, xMax, pad.left, pad.left + innerW);
  const py = (y: number) => scaleLinear(y, domain.min, domain.max, pad.top + innerH, pad.top);
  const xPositions = xs.map(px);
  const baseY = pad.top + innerH;
  const showMarker = marker !== null && marker.x >= xMin && marker.x <= xMax;

  const mapped = series.map((s) => ({
    ...s,
    px: s.points.map((p) => ({
      x: px(p.x),
      y: Number.isFinite(p.y as number) ? py(p.y as number) : null,
    })),
    byX: new Map(
      s.points.filter((p) => Number.isFinite(p.y as number)).map((p) => [p.x, p.y as number] as const),
    ),
  }));

  const activeX = active !== null && active < xs.length ? xs[active] : null;
  const rows =
    activeX === null
      ? []
      : mapped.map((s) => {
          const v = s.byX.get(activeX);
          return { key: s.key, name: s.name, color: s.color, value: v === undefined ? null : v };
        });
  const valueText = (v: number | null) => (v === null ? c.noValue : formatValue(v));
  const announcement =
    activeX === null
      ? ""
      : `${c.epoch(activeX)}: ${rows.map((r) => `${r.name} ${valueText(r.value)}`).join(", ")}`;

  const setFromPointer = (e: PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width === 0) return;
    const i = nearestIndex(xPositions, ((e.clientX - rect.left) / rect.width) * width);
    setActive(i < 0 ? null : i);
  };
  const onKeyDown = (e: KeyboardEvent<SVGSVGElement>) => {
    const last = xs.length - 1;
    const from = active ?? last;
    let next: number | null | undefined;
    if (e.key === "ArrowRight") next = Math.min(last, active === null ? last : from + 1);
    else if (e.key === "ArrowLeft") next = Math.max(0, active === null ? last : from - 1);
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = last;
    else if (e.key === "Escape") next = null;
    if (next === undefined) return;
    e.preventDefault();
    setActive(next);
  };

  // The card is about as wide as its longest row; flipping near the right edge
  // only needs that estimate, and the card itself never intercepts the pointer.
  const longest = Math.max(
    c.epoch(activeX ?? 0).length,
    ...rows.map((r) => r.name.length + valueText(r.value).length + 2),
  );
  const tipW = Math.min(Math.ceil(longest * 7.2) + 36, Math.max(120, width - 8));
  const tipH = 30 + rows.length * 19;
  const tip =
    activeX === null
      ? null
      : placeTooltip({
          anchorX: px(activeX),
          anchorY: pad.top + innerH / 2,
          width: tipW,
          height: tipH,
          chartWidth: width,
          chartHeight: height,
        });

  const single = mapped.length === 1;
  const yTicks = domain.ticks;

  return (
    <div ref={box} style={{ position: "relative", width: "100%" }}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width={width}
        height={height}
        role="img"
        aria-label={title}
        aria-describedby={hintId}
        tabIndex={0}
        onPointerDown={setFromPointer}
        onPointerMove={setFromPointer}
        onPointerLeave={(e) => {
          if (e.pointerType === "mouse") setActive(null);
        }}
        onBlur={() => setActive(null)}
        onKeyDown={onKeyDown}
        style={{ display: "block", maxWidth: "100%", touchAction: "pan-y", outlineOffset: 2 }}
      >
        {yTicks.map((tick, i) => (
          <g key={tick}>
            <line
              x1={pad.left}
              x2={width - pad.right}
              y1={py(tick)}
              y2={py(tick)}
              stroke="var(--vf-panel-stroke)"
              strokeOpacity={0.6}
              strokeWidth={1}
            />
            <text
              x={pad.left - 8}
              y={py(tick) + FONT * 0.35}
              fontSize={FONT}
              fontFamily={MONO}
              fill="var(--vf-text-muted)"
              textAnchor="end"
            >
              {tickLabels[i]}
            </text>
          </g>
        ))}
        <line x1={pad.left} x2={width - pad.right} y1={baseY} y2={baseY} stroke="var(--vf-panel-stroke)" />
        {xTicks(xMin, xMax, Math.max(2, Math.floor(innerW / 64))).map((tick) => (
          <text
            key={tick}
            x={px(tick)}
            y={height - 7}
            fontSize={FONT}
            fontFamily={MONO}
            fill="var(--vf-text-muted)"
            textAnchor={tick === xMin && xMin !== xMax ? "start" : tick === xMax && xMin !== xMax ? "end" : "middle"}
          >
            {tick}
          </text>
        ))}
        {showMarker && marker && (
          <g>
            <line
              data-best-epoch={marker.x}
              x1={px(marker.x)}
              x2={px(marker.x)}
              y1={pad.top}
              y2={baseY}
              stroke={marker.color}
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            {marker.label && (
              <text
                x={px(marker.x) + (px(marker.x) > width - pad.right - 60 ? -4 : 4)}
                y={pad.top + FONT}
                fontSize={FONT}
                fontFamily={MONO}
                fill={marker.color}
                textAnchor={px(marker.x) > width - pad.right - 60 ? "end" : "start"}
              >
                {marker.label}
              </text>
            )}
          </g>
        )}
        {single && (
          <path d={buildAreaPath(mapped[0].px, baseY)} fill={mapped[0].color} fillOpacity={0.07} stroke="none" />
        )}
        {mapped.map((s) => {
          const lastRun = [...s.points].reverse().find((p) => Number.isFinite(p.y as number));
          return (
            <g key={s.key}>
              <path
                d={buildPath(s.px)}
                fill="none"
                stroke={s.color}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
              {isolatedPoints(s.px).map((p) => (
                <circle key={p.x} cx={p.x} cy={p.y as number} r={3} fill={s.color} />
              ))}
              {lastRun && (
                <circle cx={px(lastRun.x)} cy={py(lastRun.y as number)} r={3.5} fill={s.color} />
              )}
            </g>
          );
        })}
        {activeX !== null && (
          <g pointerEvents="none">
            <line
              data-guide={activeX}
              x1={px(activeX)}
              x2={px(activeX)}
              y1={pad.top}
              y2={baseY}
              stroke="var(--vf-text-dim)"
              strokeWidth={1}
            />
            {mapped.map((s) => {
              const v = s.byX.get(activeX);
              return v === undefined ? null : (
                <circle
                  key={s.key}
                  cx={px(activeX)}
                  cy={py(v)}
                  r={4.5}
                  fill={s.color}
                  stroke="rgb(var(--lt-surface, 10 12 16))"
                  strokeWidth={2}
                />
              );
            })}
          </g>
        )}
      </svg>
      <span id={hintId} style={visuallyHidden}>
        {c.keyboardHint}
      </span>
      <div role="status" aria-live="polite" style={visuallyHidden}>
        {announcement}
      </div>
      {tip && activeX !== null && (
        <div
          data-tooltip={activeX}
          style={{
            position: "absolute",
            left: tip.left,
            top: tip.top,
            width: tipW,
            boxSizing: "border-box",
            padding: "8px 10px",
            background: "rgb(var(--lt-surface, 10 12 16) / 0.94)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 8,
            boxShadow: "0 6px 20px rgb(0 0 0 / 0.25)",
            color: "var(--vf-text)",
            fontFamily: MONO,
            fontSize: FONT + 1,
            lineHeight: "19px",
            pointerEvents: "none",
            zIndex: 2,
          }}
        >
          <div style={{ color: "var(--vf-text-dim)", fontSize: FONT, marginBottom: 2 }}>
            {c.epoch(activeX)}
          </div>
          {rows.map((r) => (
            <div key={r.key} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span
                aria-hidden
                style={{ width: 10, height: 10, borderRadius: 3, background: r.color, flex: "none" }}
              />
              <span
                style={{
                  flex: 1,
                  minWidth: 0,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  color: "var(--vf-text-dim)",
                }}
              >
                {r.name}
              </span>
              <span style={{ color: r.value === null ? "var(--vf-text-muted)" : "var(--vf-text)" }}>
                {valueText(r.value)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
