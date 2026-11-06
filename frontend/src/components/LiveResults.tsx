/** The "Resultados" tab of the training sheet: the curves of the training in
 *  flight, the epoch its saved checkpoint comes from and, in a multi-unit job,
 *  the units already closed. Everything is read off the events the sheet
 *  receives (lib/live-results.ts). */

import { artifactUrl } from "../api/client";
import { useT } from "../i18n/useT";
import {
  bestEpoch,
  bestRule,
  curveOf,
  liveUnits,
  metricKey,
  formatValue,
  latestConfusion,
  overridesSummary,
  type BestEpoch,
  type CurvePoint,
  type InsightEvent,
} from "../lib/live-results";
import type { StorySpec } from "../lib/story-family";
import { overfittingSince } from "../lib/training-story";
import type { LiveConfusion, TrainingEvent } from "../types/run";

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
  const confusion = latestConfusion(epochs);

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
      <LiveChart
        title={r.lrChart}
        series={[{ name: r.lrChart, color: TRAIN_COLOR, points: curveOf(epochs, "lr") }]}
        best={best}
        accent={accent}
      />
      {current?.insight && <GradCamGallery insight={current.insight} accent={accent} />}
      {confusion && (
        <ConfusionView epoch={confusion.epoch} confusion={confusion.confusion} accent={accent} />
      )}
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

const caption = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  letterSpacing: "0.12em",
  textTransform: "uppercase" as const,
  color: "var(--vf-text-muted)",
  marginBottom: 6,
};

/** The latest epoch's validation confusion: the matrix, its cells shaded by the
 *  share of their row, or the most frequent mistakes when there are too many
 *  classes for a grid. */
function ConfusionView({
  epoch,
  confusion,
  accent,
}: {
  epoch: number;
  confusion: LiveConfusion;
  accent: string;
}) {
  const r = useT().liveResults;
  const { classes, matrix, top } = confusion;
  const box = {
    margin: 0,
    padding: 12,
    background: "rgb(var(--lt-ink, 255 255 255) / 0.025)",
    border: "1px solid var(--vf-panel-stroke)",
    borderRadius: 12,
  };

  if (matrix) {
    return (
      <figure style={box} data-confusion="matrix">
        <figcaption style={caption}>
          <span style={{ color: accent }}>//</span> {r.confusionTitle(epoch)}
          <div style={{ letterSpacing: "0.04em", textTransform: "none", marginTop: 2 }}>
            {r.confusionAxes}
          </div>
        </figcaption>
        <div style={{ overflowX: "auto" }}>
          <table style={{ borderCollapse: "collapse", fontSize: 11, fontFamily: "var(--font-mono)" }}>
            <thead>
              <tr>
                <th />
                {classes.map((name) => (
                  <th key={name} title={name} style={headCell}>
                    {name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row, i) => {
                const total = row.reduce((sum, n) => sum + n, 0);
                return (
                  <tr key={classes[i]}>
                    <th title={classes[i]} style={{ ...headCell, textAlign: "right" }}>
                      {classes[i]}
                    </th>
                    {row.map((count, j) => {
                      const share = total > 0 ? count / total : 0;
                      return (
                        <td
                          key={classes[j]}
                          title={`${classes[i]} → ${classes[j]}: ${count}`}
                          style={{
                            minWidth: 28,
                            padding: "4px 6px",
                            textAlign: "center",
                            color: share > 0.5 ? "var(--vf-text)" : "var(--vf-text-dim)",
                            background: `color-mix(in oklch, ${i === j ? accent : "oklch(0.7 0.18 22)"} ${Math.round(share * 70)}%, transparent)`,
                            border: "1px solid var(--vf-panel-stroke)",
                          }}
                        >
                          {count}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </figure>
    );
  }

  if (!top || top.length === 0) return null;
  return (
    <figure style={box} data-confusion="top">
      <figcaption style={caption}>
        <span style={{ color: accent }}>//</span> {r.topTitle(epoch)}
      </figcaption>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <tbody>
          {top.map(([actual, predicted, count]) => (
            <tr key={`${actual}-${predicted}`} style={{ color: "var(--vf-text-dim)" }}>
              <td style={cell}>{r.topRow(classes[actual] ?? String(actual), classes[predicted] ?? String(predicted))}</td>
              <td style={{ ...cell, textAlign: "right", fontFamily: "var(--font-mono)" }}>{count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}

/** The latest Grad-CAM of the unit: the same validation images each time, so
 *  two looks at the tab compare like with like. */
function GradCamGallery({ insight, accent }: { insight: InsightEvent; accent: string }) {
  const r = useT().liveResults;
  return (
    <figure
      data-insight="gradcam"
      style={{
        margin: 0,
        padding: 12,
        background: "rgb(var(--lt-ink, 255 255 255) / 0.025)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 12,
      }}
    >
      <figcaption style={caption}>
        <span style={{ color: accent }}>//</span> {r.gradcamTitle(insight.epoch, insight.best)}
        <div style={{ letterSpacing: "0.04em", textTransform: "none", marginTop: 2 }}>
          {r.gradcamNote}
        </div>
      </figcaption>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(96px, 1fr))",
          gap: 8,
        }}
      >
        {insight.images.map((image) => {
          const label = r.gradcamCaption(image.true_class, image.predicted_class);
          const wrong = image.true_class !== image.predicted_class;
          return (
            <div key={image.path} style={{ display: "grid", gap: 4 }}>
              {/* The file is rewritten in place each time: the epoch keeps the
                  browser from showing the previous one from its cache. */}
              <img
                src={`${artifactUrl(image.path)}?epoch=${insight.epoch}`}
                alt={label}
                style={{ width: "100%", aspectRatio: "1", objectFit: "cover", borderRadius: 6 }}
              />
              <span
                style={{
                  fontSize: 11,
                  color: wrong ? "oklch(0.74 0.18 22)" : "var(--vf-text-dim)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
                title={label}
              >
                {label}
              </span>
            </div>
          );
        })}
      </div>
    </figure>
  );
}

const headCell = {
  maxWidth: 90,
  padding: "2px 6px",
  overflow: "hidden",
  textOverflow: "ellipsis",
  whiteSpace: "nowrap" as const,
  color: "var(--vf-text-muted)",
  fontWeight: 400,
};

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
