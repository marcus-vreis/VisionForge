/** The "Resultados" tab of the training sheet: the curves of the training in
 *  flight, the epoch its saved checkpoint comes from and, in a multi-unit job,
 *  the units already closed. Everything is read off the events the sheet
 *  receives (lib/live-results.ts). */

import { useT } from "../i18n/useT";
import {
  bestEpoch,
  bestRule,
  curveOf,
  liveUnits,
  metricKey,
  formatValue,
  overridesSummary,
  type BestEpoch,
  type CurvePoint,
} from "../lib/live-results";
import type { StorySpec } from "../lib/story-family";
import { overfittingSince } from "../lib/training-story";
import type { TrainingEvent } from "../types/run";

const TRAIN_COLOR = "oklch(calc(0.74 - var(--lt-flip, 0) * 0.18) 0.16 240)";
const VAL_COLOR = "oklch(calc(0.84 - var(--lt-flip, 0) * 0.38) 0.18 75)";

interface LiveResultsProps {
  events: readonly TrainingEvent[];
  spec: StorySpec;
  accent: string;
}

export function LiveResults({ events, spec, accent }: LiveResultsProps) {
  const t = useT();
  const r = t.liveResults;
  const units = liveUnits(events);
  const current = units.at(-1);
  const epochs = current?.epochs ?? [];

  if (epochs.length === 0) {
    return (
      <p style={{ margin: 0, padding: 16, color: "var(--vf-text-muted)", fontSize: 13 }}>
        {spec.family === "patchcore" ? r.noEpochsPatchcore : r.empty}
      </p>
    );
  }

  const label = (key: string) =>
    key in t.compareRuns.curves
      ? t.compareRuns.curves[key as keyof typeof t.compareRuns.curves]
      : key;
  const best = bestEpoch(epochs, bestRule(spec, epochs));
  const overfit = overfittingSince(epochs);
  const metric = metricKey(spec.metric);
  const closed = units.filter((u) => u.closed);
  const multiUnit = units.some((u) => u.index !== null);

  return (
    <div style={{ display: "grid", gap: 12 }}>
      {best && (
        <div
          style={{
            padding: "10px 14px",
            border: `1px solid ${accent}`,
            borderRadius: 12,
            fontSize: 13,
            color: "var(--vf-text)",
          }}
        >
          {r.best(best.epoch, label(best.key), formatValue(best.value))}
          <span style={{ color: "var(--vf-text-muted)" }}>
            {" "}
            · {t.compareRuns.direction[best.direction]} · {r.bestNote}
          </span>
        </div>
      )}
      {overfit !== null && (
        <div role="note" style={{ fontSize: 12, color: "var(--vf-text-dim)" }}>
          {t.trainingStory.overfitting(overfit)}
        </div>
      )}
      <LiveChart
        title={r.lossChart}
        series={[
          { name: label("train_loss"), color: TRAIN_COLOR, points: curveOf(epochs, "train_loss") },
          { name: label("val_loss"), color: VAL_COLOR, points: curveOf(epochs, "val_loss") },
        ]}
        best={best}
        accent={accent}
      />
      <LiveChart
        title={label(metric)}
        series={[{ name: label(metric), color: VAL_COLOR, points: curveOf(epochs, metric) }]}
        best={best}
        accent={accent}
      />
      {multiUnit && closed.length > 0 && (
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <caption
            style={{
              textAlign: "left",
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              color: "var(--vf-text-muted)",
              paddingBottom: 6,
            }}
          >
            {r.unitsTitle}
          </caption>
          <thead>
            <tr style={{ color: "var(--vf-text-muted)", textAlign: "left" }}>
              <th style={cell}>#</th>
              <th style={cell}>{r.changesCol}</th>
              <th style={cell}>{r.epochsCol}</th>
              <th style={cell}>{r.bestCol}</th>
            </tr>
          </thead>
          <tbody>
            {closed.map((u) => {
              const unitBest = bestEpoch(u.epochs, bestRule(spec, u.epochs));
              return (
                <tr key={u.index ?? 0} style={{ color: "var(--vf-text-dim)" }}>
                  <td style={cell}>{(u.index ?? 0) + 1}</td>
                  <td style={{ ...cell, fontFamily: "var(--font-mono)", wordBreak: "break-all" }}>
                    {overridesSummary(u.overrides) || "—"}
                  </td>
                  <td style={cell}>{u.epochs.length}</td>
                  <td style={cell}>
                    {unitBest
                      ? r.bestShort(unitBest.epoch, label(unitBest.key), formatValue(unitBest.value))
                      : "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

const cell = { padding: "4px 8px 4px 0", borderBottom: "1px solid var(--vf-panel-stroke)" };

interface ChartSeries {
  name: string;
  color: string;
  points: CurvePoint[];
}

/** One small chart of the epochs so far. A series the trainer did not report
 *  is left out, and a chart with none is not drawn. */
function LiveChart({
  title,
  series: all,
  best,
  accent,
}: {
  title: string;
  series: ChartSeries[];
  best: BestEpoch | null;
  accent: string;
}) {
  const series = all.filter((s) => s.points.length > 0);
  if (series.length === 0) return null;

  const width = 640;
  const height = 150;
  const pad = { top: 10, right: 12, bottom: 22, left: 52 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const xs = series.flatMap((s) => s.points.map((p) => p.epoch));
  const ys = series.flatMap((s) => s.points.map((p) => p.value));
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const x = (epoch: number) => pad.left + ((epoch - xMin) / Math.max(xMax - xMin, 1)) * innerW;
  // A number that never moved (an accuracy stuck at 1.0) is a level line through
  // the middle, with one label: two equal labels at the edges read as a range.
  const flat = yMax === yMin;
  const y = (value: number) =>
    flat ? pad.top + innerH / 2 : pad.top + innerH - ((value - yMin) / (yMax - yMin)) * innerH;
  const showBest = best !== null && best.epoch >= xMin && best.epoch <= xMax;

  return (
    <figure
      style={{
        margin: 0,
        padding: 12,
        background: "rgb(var(--lt-ink, 255 255 255) / 0.025)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 12,
      }}
    >
      <figcaption
        style={{
          display: "flex",
          gap: 12,
          flexWrap: "wrap",
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.12em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
          marginBottom: 6,
        }}
      >
        <span>
          <span style={{ color: accent }}>//</span> {title}
        </span>
        {series.length > 1 &&
          series.map((s) => (
            <span key={s.name} style={{ color: s.color }}>
              ● {s.name}
            </span>
          ))}
      </figcaption>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        role="img"
        aria-label={title}
        style={{ display: "block" }}
      >
        {flat ? (
          <text x={pad.left - 6} y={y(yMin) + 3} fontSize="10" fill="var(--vf-text-muted)" textAnchor="end">
            {formatValue(yMin)}
          </text>
        ) : (
          <>
            <text x={pad.left - 6} y={pad.top + 8} fontSize="10" fill="var(--vf-text-muted)" textAnchor="end">
              {formatValue(yMax)}
            </text>
            <text x={pad.left - 6} y={pad.top + innerH} fontSize="10" fill="var(--vf-text-muted)" textAnchor="end">
              {formatValue(yMin)}
            </text>
          </>
        )}
        <text x={pad.left} y={height - 6} fontSize="10" fill="var(--vf-text-muted)">
          {xMin}
        </text>
        <text x={width - pad.right} y={height - 6} fontSize="10" fill="var(--vf-text-muted)" textAnchor="end">
          {xMax}
        </text>
        <line
          x1={pad.left}
          y1={pad.top + innerH}
          x2={width - pad.right}
          y2={pad.top + innerH}
          stroke="var(--vf-panel-stroke)"
        />
        {showBest && (
          <line
            data-best-epoch={best.epoch}
            x1={x(best.epoch)}
            y1={pad.top}
            x2={x(best.epoch)}
            y2={pad.top + innerH}
            stroke={accent}
            strokeDasharray="3 3"
          />
        )}
        {series.map((s) => (
          <g key={s.name}>
            <path
              d={s.points.map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.epoch)} ${y(p.value)}`).join(" ")}
              fill="none"
              stroke={s.color}
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {/* A single epoch is a point, which a line does not draw. */}
            {s.points.length === 1 && (
              <circle cx={x(s.points[0].epoch)} cy={y(s.points[0].value)} r={3.5} fill={s.color} />
            )}
          </g>
        ))}
      </svg>
    </figure>
  );
}
