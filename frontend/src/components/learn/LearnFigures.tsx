/** The figures of the "Começando do zero" screen (ADR-207).
 *
 * Three kinds, and the screen says which is which: the network drawing is the
 * one the training sheet shows (components/training-story); the split, batch,
 * curve and matrix figures are drawn from the numbers in lib/learn.ts and are
 * labelled illustrative; the screens are images captured from this interface by
 * scripts/capture_docs_images.py.
 */

import { useEffect, useState, type CSSProperties, type ReactNode } from "react";

import { useT } from "../../i18n/useT";
import {
  CONFUSION_EXAMPLE,
  LEARNING_RATE_CHOICES,
  accuracyOf,
  bestEpoch,
  lossCurve,
  splitParts,
  type LearningRateChoice,
  type SplitPart,
} from "../../lib/learn";
import { activeFlows, storyMotion } from "../../lib/training-story";
import { StoryDrawing } from "../training-story/drawings";
import { LegendGlyph } from "../training-story/flow";
import { accentForStory, storyPalette } from "../training-story/tone";
import {
  useDocumentVisible,
  usePrefersReducedMotion,
  useResolvedTheme,
} from "../training-story/use-viewing";
import { LineChart } from "../ui/LineChart";

const MONO: CSSProperties = { fontFamily: "var(--font-mono)" };

/** The frame every figure sits in: a panel, the figure, one caption line. */
export function FigureFrame({
  children,
  caption,
  note,
}: {
  children: ReactNode;
  caption: string;
  /** A second, dimmer line: "illustrative figure". */
  note?: string;
}) {
  return (
    <figure className="vf-learn-figure">
      <div className="vf-learn-figure-body">{children}</div>
      <figcaption>
        {caption}
        {note && <span className="vf-learn-figure-note">{note}</span>}
      </figcaption>
    </figure>
  );
}

/** How long the drawing waits between two validation pulses. */
const EPOCH_MS = 5_000;

/** The network drawing of the training sheet, running on its own. */
export function TrainingLoopFigure() {
  const c = useT().learn.chapters.idea;
  const theme = useResolvedTheme();
  const reducedMotion = usePrefersReducedMotion();
  const documentVisible = useDocumentVisible();
  const [epoch, setEpoch] = useState(0);
  useEffect(() => {
    const id = window.setInterval(() => setEpoch((n) => n + 1), EPOCH_MS);
    return () => window.clearInterval(id);
  }, []);

  const palette = storyPalette(theme);
  const accent = accentForStory("classification", undefined, theme);
  const legend = [
    { kind: "dot", color: accent, text: c.legend.data },
    { kind: "ring", color: palette.gradient, text: c.legend.gradient },
    { kind: "diamond", color: palette.validation, text: c.legend.validation },
  ] as const;

  return (
    <FigureFrame caption={c.caption}>
      <StoryDrawing
        family="cnn"
        output="classes"
        phase={null}
        accent={accent}
        stage="training"
        flows={activeFlows("training", "cnn")}
        motion={storyMotion({
          stage: "training",
          reducedMotion,
          documentVisible,
          sheetVisible: true,
        })}
        pulseKey={epoch}
        label={c.figureLabel}
        theme={theme}
      />
      <div className="vf-learn-legend">
        {legend.map((item) => (
          <span key={item.kind}>
            <LegendGlyph kind={item.kind} color={item.color} />
            {item.text}
          </span>
        ))}
      </div>
    </FigureFrame>
  );
}

const SPLIT_IMAGES = 40;
const SPLIT_ORDER: readonly SplitPart[] = ["train", "val", "test"];

/** Forty tiles in three groups: what the model studies, the quiz, the exam. */
export function SplitFigure() {
  const t = useT().learn;
  const c = t.chapters.data;
  const parts = splitParts(SPLIT_IMAGES);
  return (
    <FigureFrame caption={c.caption} note={t.illustrative}>
      <div className="vf-learn-split" role="img" aria-label={c.figureLabel}>
        {SPLIT_ORDER.map((part) => {
          const count = parts.filter((p) => p === part).length;
          return (
            <div key={part} className="vf-learn-split-part" data-part={part} style={{ flexGrow: count }}>
              <div className="vf-learn-split-tiles">
                {Array.from({ length: count }, (_, i) => (
                  <span key={i} style={{ animationDelay: `${i * 18}ms` }} />
                ))}
              </div>
              <div className="vf-learn-split-name">
                {c.parts[part]} · {Math.round((count / SPLIT_IMAGES) * 100)}%
              </div>
              <div className="vf-learn-split-note">{c.partNotes[part]}</div>
            </div>
          );
        })}
      </div>
    </FigureFrame>
  );
}

const BATCHES = 4;
const BATCH_IMAGES = 8;
/** Seconds each batch stays lit; the loop is one epoch plus a pause. */
const BATCH_SECONDS = 1.1;

/** One epoch: the batches light up in turn, then the epoch is counted. */
export function BatchesFigure() {
  const c = useT().learn.chapters.loop;
  const loop = `${(BATCHES + 1.5) * BATCH_SECONDS}s`;
  return (
    <FigureFrame caption={c.caption}>
      <div className="vf-learn-batches" role="img" aria-label={c.figureLabel}>
        {Array.from({ length: BATCHES }, (_, b) => (
          <div
            key={b}
            className="vf-learn-batch"
            style={{ animationDuration: loop, animationDelay: `${b * BATCH_SECONDS}s` }}
          >
            <div className="vf-learn-batch-tiles">
              {Array.from({ length: BATCH_IMAGES }, (_, i) => (
                <span key={i} />
              ))}
            </div>
            <div style={{ ...MONO, fontSize: 11 }}>{c.batch(b + 1)}</div>
          </div>
        ))}
        <div
          className="vf-learn-batch vf-learn-batch--done"
          style={{ animationDuration: loop, animationDelay: `${BATCHES * BATCH_SECONDS}s` }}
        >
          <div style={{ ...MONO, fontSize: 11 }}>{c.epochDone}</div>
        </div>
      </div>
    </FigureFrame>
  );
}

/** Loss per epoch at three learning rates, on the chart the live results use. */
export function CurvesFigure({ initial = "good" }: { initial?: LearningRateChoice }) {
  const t = useT().learn;
  const c = t.chapters.curves;
  const theme = useResolvedTheme();
  const [rate, setRate] = useState<LearningRateChoice>(initial);
  const points = lossCurve(rate);
  const best = bestEpoch(points);
  const palette = storyPalette(theme);
  const trainColor = accentForStory("classification", undefined, theme);
  const note = rate === "good" ? c.notes.good(best) : c.notes[rate];

  return (
    <FigureFrame caption={note} note={t.illustrative}>
      <div className="vf-learn-rate" role="group" aria-label={c.rateLabel}>
        <span style={{ ...MONO, fontSize: 11, color: "var(--vf-text-muted)" }}>{c.rateLabel}</span>
        {LEARNING_RATE_CHOICES.map((choice) => (
          <button
            key={choice}
            type="button"
            aria-pressed={choice === rate}
            onClick={() => setRate(choice)}
          >
            {c.rates[choice]}
          </button>
        ))}
      </div>
      <LineChart
        title={c.figureLabel}
        height={240}
        formatValue={(value) => value.toFixed(2)}
        series={[
          {
            key: "train",
            name: c.train,
            color: trainColor,
            points: points.map((p) => ({ x: p.epoch, y: p.train })),
          },
          {
            key: "val",
            name: c.val,
            color: palette.validation,
            points: points.map((p) => ({ x: p.epoch, y: p.val })),
          },
        ]}
        marker={rate === "good" ? { x: best, color: palette.gradient, label: c.best(best) } : null}
      />
      <div className="vf-learn-legend">
        {[
          { text: c.train, color: trainColor },
          { text: c.val, color: palette.validation },
        ].map((item) => (
          <span key={item.text}>
            <i className="vf-learn-swatch" style={{ background: item.color }} aria-hidden="true" />
            {item.text}
          </span>
        ))}
      </div>
    </FigureFrame>
  );
}

/** A 3 × 3 confusion matrix: the diagonal in the accent, the mistakes in ink. */
export function ConfusionFigure() {
  const t = useT();
  const c = t.learn.chapters.result;
  const matrix = CONFUSION_EXAMPLE;
  const total = matrix.flat().reduce((sum, n) => sum + n, 0);
  const correct = matrix.reduce((sum, row, i) => sum + row[i], 0);
  const peak = Math.max(...matrix.flat());
  const percent = `${Math.round(accuracyOf(matrix) * 100)}%`;

  return (
    <FigureFrame caption={c.caption} note={t.learn.illustrative}>
      <div className="vf-learn-matrix-wrap">
        <table className="vf-learn-matrix" aria-label={c.figureLabel}>
          <thead>
            <tr>
              <td />
              <th colSpan={matrix.length} scope="colgroup">
                {c.predictedAxis}
              </th>
            </tr>
            <tr>
              <th scope="col">{c.trueAxis}</th>
              {c.classes.map((name) => (
                <th key={name} scope="col">
                  {name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={c.classes[i]}>
                <th scope="row">{c.classes[i]}</th>
                {row.map((count, j) => (
                  <td
                    key={c.classes[j]}
                    data-hit={i === j ? "" : undefined}
                    style={{ "--share": count / peak } as CSSProperties}
                  >
                    {count}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        <p style={{ ...MONO, fontSize: 12, margin: 0, color: "var(--vf-text-dim)" }}>
          {c.accuracy(correct, total, percent)}
        </p>
      </div>
    </FigureFrame>
  );
}

/** A screen of this interface, in a window frame so it reads as a picture. */
export function ScreenFigure({
  src,
  alt,
  caption,
  width,
  height,
}: {
  src: string;
  alt: string;
  caption: string;
  width: number;
  height: number;
}) {
  return (
    <figure className="vf-learn-figure vf-learn-figure--screen">
      <div className="vf-learn-window">
        <div className="vf-learn-window-bar" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <img src={src} alt={alt} width={width} height={height} loading="lazy" decoding="async" />
      </div>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}
