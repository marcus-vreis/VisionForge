/** The "Resultados" tab of the training sheet: the curves of the training in
 *  flight, the epoch its saved checkpoint comes from and, in a multi-unit job,
 *  the units already closed. Everything is read off the events the sheet
 *  receives (lib/live-results.ts). */

import { artifactUrl } from "../api/client";
import { LineChart } from "./ui/LineChart";
import { TRAIN_COLOR, VAL_COLOR } from "../lib/run-curves";
import { useT } from "../i18n/useT";
import {
  bestEpoch,
  bestRule,
  confusionAt,
  curveOf,
  liveUnits,
  metricKey,
  formatValue,
  latestConfusion,
  macroF1Curve,
  overridesSummary,
  recallCurves,
  type BestEpoch,
  type CurvePoint,
  type InsightEvent,
} from "../lib/live-results";
import type { StorySpec } from "../lib/story-family";
import { overfittingSince } from "../lib/training-story";
import type { LiveConfusion, TrainingEvent } from "../types/run";


/** More recall lines than this cannot be told apart in one chart. */
const RECALL_LINES = 8;

/** One hue per class, spread around the wheel; the lightness follows the theme
 *  as the train and validation colors do. */
function classColor(index: number, classes: number): string {
  const hue = Math.round((index * 360) / Math.max(classes, 1) + 30) % 360;
  return `oklch(calc(0.8 - var(--lt-flip, 0) * 0.3) 0.15 ${hue})`;
}

/** Charts and matrices sit side by side on the wide sheet and stack on a
 *  narrow window. */
const sideBySide = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(min(320px, 100%), 1fr))",
  gap: 12,
  alignItems: "start",
};

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
  const latestIsBest = confusion !== null && best !== null && confusion.epoch === best.epoch;
  // The matrix of the epoch the saved checkpoint comes from, when it is not
  // the one already on screen.
  const bestMatrix = best && !latestIsBest ? confusionAt(epochs, best.epoch) : null;
  const recall = recallCurves(epochs, RECALL_LINES);
  const classCount = confusion?.confusion.classes.length ?? 0;

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
      <div style={sideBySide}>
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
          title={r.f1Chart}
          series={[{ name: r.f1Chart, color: VAL_COLOR, points: macroF1Curve(epochs) }]}
          best={best}
          accent={accent}
        />
        <LiveChart
          title={r.recallChart}
          series={recall.series.map((c) => ({
            name: c.name,
            color: classColor(c.index, classCount),
            points: c.points,
          }))}
          note={recall.omitted > 0 ? r.recallOmitted(recall.series.length, recall.omitted) : undefined}
          best={best}
          accent={accent}
        />
        <LiveChart
          title={r.lrChart}
          series={[{ name: r.lrChart, color: TRAIN_COLOR, points: curveOf(epochs, "lr") }]}
          best={best}
          accent={accent}
        />
      </div>
      {current?.insight && <GradCamGallery insight={current.insight} accent={accent} />}
      {confusion && (
        <div style={sideBySide}>
          <ConfusionView
            title={
              latestIsBest && confusion.confusion.matrix
                ? r.confusionLatestIsBest(confusion.epoch)
                : undefined
            }
            epoch={confusion.epoch}
            confusion={confusion.confusion}
            accent={accent}
          />
          {best && bestMatrix?.matrix && (
            <ConfusionView
              title={r.confusionBestTitle(best.epoch)}
              epoch={best.epoch}
              confusion={bestMatrix}
              accent={accent}
            />
          )}
        </div>
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

/** One epoch's validation confusion: the matrix, its cells shaded by the share
 *  of their row, or the most frequent mistakes when there are too many classes
 *  for a grid. `title` replaces the matrix's "epoch N" heading. */
function ConfusionView({
  title,
  epoch,
  confusion,
  accent,
}: {
  title?: string;
  epoch: number;
  confusion: LiveConfusion;
  accent: string;
}) {
  const r = useT().liveResults;
  const { classes, matrix, top } = confusion;
  const box = {
    margin: 0,
    padding: 12,
    // A grid item is as wide as its content unless told otherwise, and a wide
    // matrix would push the sheet: it scrolls inside its own box instead.
    minWidth: 0,
    background: "rgb(var(--lt-ink, 255 255 255) / 0.025)",
    border: "1px solid var(--vf-panel-stroke)",
    borderRadius: 12,
  };

  if (matrix) {
    return (
      <figure style={box} data-confusion="matrix">
        <figcaption style={caption}>
          <span style={{ color: accent }}>//</span> {title ?? r.confusionTitle(epoch)}
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
  note,
  best,
  accent,
}: {
  title: string;
  series: ChartSeries[];
  /** A line under the legend, e.g. which series were left out. */
  note?: string;
  best: BestEpoch | null;
  accent: string;
}) {
  const t = useT();
  const series = all.filter((s) => s.points.length > 0);
  if (series.length === 0) return null;

  return (
    <figure
      style={{
        margin: 0,
        padding: 12,
        minWidth: 0,
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
          fontSize: 11,
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
        {note && (
          <span style={{ flexBasis: "100%", letterSpacing: "0.04em", textTransform: "none" }}>
            {note}
          </span>
        )}
      </figcaption>
      <LineChart
        title={title}
        height={170}
        formatValue={formatValue}
        marker={best ? { x: best.epoch, color: accent, label: t.chart.best } : null}
        series={series.map((s) => ({
          key: s.name,
          name: s.name,
          color: s.color,
          points: s.points.map((p) => ({ x: p.epoch, y: p.value })),
        }))}
      />
    </figure>
  );
}
