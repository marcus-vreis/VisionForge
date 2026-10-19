/** The "Como funciona" tab of the training sheet (ADR-118).
 *
 * A drawing of the kind of network that is learning, with three flows moving
 * through it, a strip that says which stage the run is in, one sentence about that
 * stage, and the numbers the trainer reported. The stage and the numbers are real
 * (`trainingStory`); the motion is not, and the caption says so.
 */

import type { CSSProperties } from "react";
import { useT } from "../../i18n/useT";
import type { UnitKind } from "../../lib/run-control";
import type { StoryFamily, StorySpec } from "../../lib/story-family";
import { phaseName } from "../../lib/training-phase";
import {
  activeFlows,
  STAGE_ORDER,
  storyMotion,
  trainingStory,
  type StoryNumbers,
  type StoryStage,
} from "../../lib/training-story";
import type { RunStatus, TrainingEvent } from "../../types/run";
import { canReplay } from "../../lib/training-replay";
import { StoryDrawing } from "./drawings";
import { LegendGlyph } from "./flow";
import {
  accentForStory,
  FAILED_COLOR,
  GRADIENT_COLOR,
  INTERRUPTED_COLOR,
  VALIDATION_COLOR,
  withAlpha,
} from "./tone";
import { useReplay } from "./use-replay";
import { useDocumentVisible, usePrefersReducedMotion } from "./use-viewing";

export interface TrainingStoryProps {
  events: readonly TrainingEvent[];
  status: RunStatus["status"];
  /** The sheet's stop verdict: a stop cut this run (`stopOutcome === "stopped"`). */
  stopped: boolean;
  /** The unit a stop lands on, which names the units of a multi-unit job. */
  unitKind: UnitKind | null;
  spec: StorySpec;
  /** The accent of the tab the run was started from, for a researcher's own task. */
  tabAccent?: string;
  /** The sheet is on screen: a minimized sheet pauses the drawing. */
  sheetVisible: boolean;
}

const MONO: CSSProperties = { fontFamily: "var(--font-mono)" };

/** Which slot of the strip a stage lights. The end states share the last one. */
function slotOf(stage: StoryStage): number {
  if (stage === "queued") return -1;
  const i = STAGE_ORDER.indexOf(stage);
  return i >= 0 ? i : STAGE_ORDER.length - 1;
}

const FAMILY_KEY = {
  cnn: "cnn",
  transformer: "transformer",
  encoder_decoder: "encoderDecoder",
  detector: "detector",
  autoencoder: "autoencoder",
  patchcore: "patchcore",
} as const satisfies Record<StoryFamily, string>;

function fmt(value: number | null): string {
  return value === null ? "—" : value.toFixed(4);
}

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ minWidth: 0 }}>
      <div
        style={{
          ...MONO,
          fontSize: 9,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
        }}
      >
        {label}
      </div>
      <div style={{ ...MONO, fontSize: 15, fontWeight: 600, color: "var(--vf-text)", marginTop: 2 }}>
        {value}
      </div>
    </div>
  );
}

function numbersOf(
  n: StoryNumbers,
  patchcore: boolean,
  t: ReturnType<typeof useT>,
  phase: string | null,
): Array<{ label: string; value: string }> {
  const words = t.trainingStory.numbers;
  const tiles: Array<{ label: string; value: string }> = [];
  if (patchcore) {
    if (phase !== null) tiles.push({ label: words.phase, value: phase });
  } else {
    tiles.push({
      label: words.epoch,
      value:
        n.epoch === null && n.totalEpochs === null
          ? "—"
          : words.epochOf(n.epoch ?? 0, n.totalEpochs),
    });
    if (n.trainLoss !== null) tiles.push({ label: words.trainLoss, value: fmt(n.trainLoss) });
    if (n.valLoss !== null) tiles.push({ label: words.valLoss, value: fmt(n.valLoss) });
  }
  if (n.metric !== null) tiles.push({ label: words.metrics[n.metric.kind], value: fmt(n.metric.value) });
  return tiles;
}

export function TrainingStory({
  events,
  status,
  stopped,
  unitKind,
  spec,
  tabAccent,
  sheetVisible,
}: TrainingStoryProps) {
  const t = useT();
  const s = t.trainingStory;
  const story = trainingStory({ events, status, stopped, unitKind, spec });
  const { family, phase } = story;
  // The replay forces a stage over the drawing and the sentence; the numbers stay the run's.
  const replay = useReplay(story.stage === "done");
  const replaying = replay.stage !== null;
  const stage = replay.stage ?? story.stage;
  const reducedMotion = usePrefersReducedMotion();
  const documentVisible = useDocumentVisible();
  const motion = storyMotion({ stage, reducedMotion, documentVisible, sheetVisible });
  const flows = activeFlows(stage, family, phase?.key ?? null);
  const accent = accentForStory(spec.task, tabAccent);
  const patchcore = family === "patchcore";

  const slot = slotOf(stage);
  const endColor =
    stage === "failed" ? FAILED_COLOR : stage === "interrupted" ? INTERRUPTED_COLOR : accent;

  // One sentence: the phase's own while PatchCore is in one, else the stage's.
  const sentence =
    patchcore && stage === "training" && phase?.key
      ? s.phases[phase.key]
      : s.stages[stage];
  const phaseText =
    phase === null ? null : `${phaseName(t, phase.label)} · ${phase.done}/${phase.total}`;
  const tiles = numbersOf(story.numbers, patchcore, t, phaseText);
  const familyName = s.families[FAMILY_KEY[family]];

  const legend = patchcore
    ? [
        { kind: "dot" as const, color: accent, text: s.legend.patchForward },
        { kind: "diamond" as const, color: VALIDATION_COLOR, text: s.legend.patchScore },
      ]
    : [
        { kind: "dot" as const, color: accent, text: s.legend.forward },
        { kind: "ring" as const, color: GRADIENT_COLOR, text: s.legend.backward },
        { kind: "diamond" as const, color: VALIDATION_COLOR, text: s.legend.validation },
      ];

  return (
    <div
      data-story-stage={stage}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 8,
        maxHeight: "min(440px, calc(100vh - 330px))",
        overflowY: "auto",
        minHeight: 0,
      }}
    >
      {/* What is drawn, and which unit of the job it is. */}
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12 }}>
        <div style={{ ...MONO, fontSize: 11, color: "var(--vf-text-dim)" }}>{familyName}</div>
        {story.unit && (
          <div
            style={{
              ...MONO,
              fontSize: 11,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: accent,
              border: `1px solid ${withAlpha(accent, 0.5)}`,
              borderRadius: 999,
              padding: "2px 10px",
              whiteSpace: "nowrap",
            }}
          >
            {s.units[story.unit.kind](story.unit.index, story.unit.total)}
          </div>
        )}
      </div>

      {/* The stages of a run, with the one it is in lit. */}
      <ol
        aria-label={s.stripLabel}
        style={{ display: "flex", gap: 6, margin: 0, padding: 0, listStyle: "none" }}
      >
        {STAGE_ORDER.map((name, i) => {
          const last = i === STAGE_ORDER.length - 1;
          const shown: StoryStage = last && slot === i ? stage : name;
          const lit = i === slot;
          const past = i < slot;
          const color = lit ? (last ? endColor : accent) : past ? accent : "var(--vf-text-muted)";
          return (
            <li
              key={name}
              aria-current={lit ? "step" : undefined}
              style={{
                ...MONO,
                flex: 1,
                textAlign: "center",
                fontSize: 10,
                letterSpacing: "0.1em",
                textTransform: "uppercase",
                padding: "5px 6px",
                borderRadius: 8,
                color: lit ? "var(--vf-text)" : color,
                border: `1px solid ${lit ? color : past ? withAlpha(accent, 0.35) : "var(--vf-panel-stroke)"}`,
                background: lit ? withAlpha(color, 0.14) : "transparent",
              }}
            >
              {s.stageNames[shown]}
            </li>
          );
        })}
      </ol>

      {/* The drawing. */}
      <div
        style={{
          background: "rgb(var(--lt-shade, 0 0 0) / 0.45)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 12,
          padding: "4px 10px",
        }}
      >
        <StoryDrawing
          family={family}
          output={spec.output}
          phase={phase}
          accent={accent}
          stage={stage}
          flows={flows}
          motion={motion}
          pulseKey={story.epochsSeen}
          label={s.figure(familyName)}
        />
      </div>

      {/* One sentence about the stage. */}
      <div
        role="status"
        aria-live="polite"
        style={{ ...MONO, fontSize: 12, lineHeight: 1.55, color: "var(--vf-text)" }}
      >
        {sentence}
      </div>

      {canReplay(story.stage, family) && (
        <button
          type="button"
          data-story-replay
          disabled={replaying}
          onClick={replay.start}
          style={{
            ...MONO,
            alignSelf: "flex-start",
            fontSize: 11,
            padding: "4px 12px",
            borderRadius: 999,
            color: "var(--vf-text)",
            background: "transparent",
            border: `1px solid ${withAlpha(accent, 0.5)}`,
            cursor: replaying ? "default" : "pointer",
            opacity: replaying ? 0.6 : 1,
          }}
        >
          {replaying ? s.replay.playing : s.replay.button}
        </button>
      )}

      {/* A sign the losses themselves show, not the drawing (ADR-119). */}
      {story.overfittingSince !== null && (
        <div
          data-story-overfitting
          style={{
            ...MONO,
            fontSize: 11.5,
            lineHeight: 1.5,
            color: "var(--vf-text)",
            borderLeft: `3px solid ${GRADIENT_COLOR}`,
            background: withAlpha(GRADIENT_COLOR, 0.1),
            borderRadius: 6,
            padding: "6px 10px",
          }}
        >
          {s.overfitting(story.overfittingSince)}
        </div>
      )}

      {/* The numbers the trainer reported. */}
      {tiles.length > 0 && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: `repeat(${tiles.length}, minmax(0, 1fr))`,
            gap: 12,
            padding: "6px 12px",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 10,
          }}
        >
          {tiles.map((tile) => (
            <Tile key={tile.label} label={tile.label} value={tile.value} />
          ))}
        </div>
      )}

      {/* The colours, and the admission that the motion is for show. */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 16px", alignItems: "center" }}>
        {legend.map((item) => (
          <span
            key={item.text}
            style={{ ...MONO, display: "inline-flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--vf-text-dim)" }}
          >
            <LegendGlyph kind={item.kind} color={item.color} />
            {item.text}
          </span>
        ))}
      </div>
      <div style={{ ...MONO, fontSize: 10.5, color: "var(--vf-text-muted)", lineHeight: 1.5 }}>
        {s.caption}
      </div>
    </div>
  );
}
