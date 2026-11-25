/** "Como funciona": the run as steps and the real code of the step it is in.
 *
 * One tab for what used to be two (the story of ADR-118 and the "Código" tab of
 * ADR-173). The left column lists the steps of the run with one plain sentence
 * each, and the live figures under the step that runs; the right column shows the
 * real code of the selected step, read from the installed package.
 *
 * While the run is going, the tab follows it: the step the event stream proves is
 * running is selected by itself (`lib/run-steps.ts`). A click pauses that, and
 * "voltar ao que está rodando" resumes it. The grain is the step, never a line.
 */

import { useReducer, type CSSProperties } from "react";
import type { CodeSettings } from "../../api/client";
import { useT } from "../../i18n/useT";
import type { UnitKind } from "../../lib/run-control";
import {
  FOLLOWING,
  followReducer,
  offPath,
  runPosition,
  stepView,
  stepWords,
  type RunStep,
} from "../../lib/run-steps";
import type { StoryFamily, StorySpec } from "../../lib/story-family";
import { phaseName } from "../../lib/training-phase";
import { activeFlows, storyMotion, trainingStory, type StoryState } from "../../lib/training-story";
import type { RunStatus, TrainingEvent } from "../../types/run";
import { StoryDrawing } from "../training-story/drawings";
import { accentForStory } from "../training-story/tone";
import { useDocumentVisible, usePrefersReducedMotion, useResolvedTheme } from "../training-story/use-viewing";
import { CodeView } from "./CodeView";
import { Figure, StepList } from "./StepList";
import { useMinWidth } from "./use-min-width";
import { useTour, type TourLoad } from "./use-tour";

const MONO: CSSProperties = { fontFamily: "var(--font-mono)" };
const BODY: CSSProperties = { margin: 0, fontSize: 13, lineHeight: 1.5 };

/** Below this width the drawing is left out: the code is what the tab is for. */
const DRAWING_MIN_WIDTH = 1100;
/** Below this width the two columns stack. */
const COLUMNS_MIN_WIDTH = 760;

const FAMILY_KEY = {
  cnn: "cnn",
  transformer: "transformer",
  encoder_decoder: "encoderDecoder",
  detector: "detector",
  autoencoder: "autoencoder",
  patchcore: "patchcore",
} as const satisfies Record<StoryFamily, string>;

export interface RunCodeProps {
  events: readonly TrainingEvent[];
  status: RunStatus["status"];
  /** The sheet's stop verdict: a stop cut this run (`stopOutcome === "stopped"`). */
  stopped: boolean;
  /** The unit a stop lands on, which names the units of a multi-unit job. */
  unitKind: UnitKind | null;
  /** The run's drawing and headline metric, recorded at submission (ADR-118). */
  spec: StorySpec;
  /** The accent of the tab the run was started from. */
  accent: string;
  /** The sheet is on screen: a minimized sheet pauses the drawing. */
  sheetVisible: boolean;
}

/** The tab, with its code loaded from the server for this run's settings. */
export function TrainingCodeTab({
  taskKey,
  settings,
  custom = false,
  ...props
}: RunCodeProps & { taskKey: string; settings: CodeSettings | null; custom?: boolean }) {
  const tour = useTour(taskKey, settings, custom);
  return <RunCodePanel {...props} tour={tour} />;
}

/** The tab for a tour in any state; the markup tests render it directly. */
export function RunCodePanel({
  events,
  status,
  stopped,
  unitKind,
  spec,
  accent,
  sheetVisible,
  tour,
}: RunCodeProps & { tour: TourLoad }) {
  const t = useT();
  const words = t.codeTour;
  const story = trainingStory({ events, status, stopped, unitKind, spec });
  const [follow, dispatch] = useReducer(followReducer, FOLLOWING);
  const sections = tour.kind === "ready" ? tour.sections : null;
  const view = stepView({ follow, position: runPosition(events, story.stage), stage: story.stage, sections });
  const wide = useMinWidth(DRAWING_MIN_WIDTH);
  const columns = useMinWidth(COLUMNS_MIN_WIDTH);
  const patchcore = story.family === "patchcore";
  const inStep = (sections ?? []).filter((s) => s.step === view.step);
  const title = (step: RunStep) => stepWords(words, step, patchcore).title;
  const scroll: CSSProperties = columns ? { minHeight: 0, overflowY: "auto" } : {};

  return (
    <div
      data-story-stage={story.stage}
      data-run-step={view.step}
      style={
        columns
          ? {
              display: "grid",
              gridTemplateColumns: `${wide ? 270 : 230}px minmax(0, 1fr)`,
              gap: 20,
              height: "max(320px, min(560px, calc(100vh - 300px)))",
            }
          : { display: "flex", flexDirection: "column", gap: 16 }
      }
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 12, ...scroll }}>
        {wide && <CompactDrawing story={story} spec={spec} tabAccent={accent} sheetVisible={sheetVisible} />}
        <StepList
          steps={view.steps}
          running={view.running}
          selected={view.step}
          accent={accent}
          patchcore={patchcore}
          figures={<Figures story={story} running={view.running} />}
          onPick={(step) => dispatch({ kind: "step", step })}
        />
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 10, minWidth: 0, ...scroll }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap", minHeight: 28 }}>
          <h3 style={{ margin: 0, fontSize: 15, fontWeight: 600, color: "var(--vf-text)" }}>{title(view.step)}</h3>
          {view.live ? (
            <span role="status" data-live style={{ ...MONO, display: "inline-flex", alignItems: "center", gap: 6, fontSize: 11, color: accent }}>
              <span aria-hidden="true" style={{ width: 7, height: 7, borderRadius: "50%", background: accent }} />
              {view.running.length > 1
                ? words.liveGroup(view.running.map((step) => title(step).toLocaleLowerCase()))
                : words.live}
            </span>
          ) : (
            view.canResume && (
              <button
                type="button"
                data-resume
                onClick={() => dispatch({ kind: "resume" })}
                style={{
                  ...MONO,
                  fontSize: 11,
                  padding: "4px 12px",
                  borderRadius: 999,
                  cursor: "pointer",
                  color: "var(--vf-text)",
                  background: "transparent",
                  border: `1px solid ${accent}`,
                }}
              >
                {words.resume}
              </button>
            )
          )}
        </div>

        {inStep.length > 1 && (
          <ul aria-label={words.functionsLabel} style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexWrap: "wrap", gap: 6 }}>
            {inStep.map((s) => {
              const picked = s.id === view.section?.id;
              return (
                <li key={s.id}>
                  <button
                    type="button"
                    aria-pressed={picked}
                    onClick={() => dispatch({ kind: "section", step: view.step, section: s.id })}
                    style={{
                      ...MONO,
                      fontSize: 12,
                      padding: "4px 10px",
                      borderRadius: 8,
                      cursor: "pointer",
                      color: "var(--vf-text)",
                      opacity: offPath(s) && !picked ? 0.6 : 1,
                      background: picked ? "var(--accent-soft)" : "transparent",
                      border: `1px solid ${picked ? accent : "var(--vf-panel-stroke)"}`,
                    }}
                  >
                    {s.name} <span style={{ fontSize: 11, color: "var(--vf-text-muted)" }}>· {s.file}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}

        {tour.kind === "loading" ? (
          <p role="status" style={BODY}>{words.loading}</p>
        ) : tour.kind === "failed" ? (
          <p role="alert" style={BODY}>{words.failed}</p>
        ) : tour.kind === "unavailable" ? (
          <p style={BODY}>{tour.reason}</p>
        ) : view.section ? (
          <CodeView section={view.section} accent={accent} />
        ) : (
          <p style={BODY}>{words.missing}</p>
        )}

        <p style={{ ...MONO, margin: 0, fontSize: 11, lineHeight: 1.5, color: "var(--vf-text-muted)" }}>{words.granularity}</p>
      </div>
    </div>
  );
}

/** What the trainer reported, under the running step: the unit of a multi-unit
 *  job, and inside an epoch its number, its losses and the headline metric (the
 *  phase, for PatchCore). Nothing that was not reported is shown. */
function Figures({ story, running }: { story: StoryState; running: readonly RunStep[] }) {
  const t = useT();
  const words = t.trainingStory.numbers;
  const { numbers: n, unit, phase } = story;
  const inEpoch = running.includes("train") || running.includes("validate");
  const items: Array<{ label: string; value: string }> = [];
  if (inEpoch && story.family === "patchcore") {
    if (phase) items.push({ label: words.phase, value: `${phaseName(t, phase.label)} · ${phase.done}/${phase.total}` });
  } else if (inEpoch) {
    items.push({
      label: words.epoch,
      value: n.epoch === null && n.totalEpochs === null ? "—" : words.epochOf(n.epoch ?? 0, n.totalEpochs),
    });
    if (n.trainLoss !== null) items.push({ label: words.trainLoss, value: n.trainLoss.toFixed(4) });
    if (n.valLoss !== null) items.push({ label: words.valLoss, value: n.valLoss.toFixed(4) });
  }
  if (inEpoch && n.metric !== null) items.push({ label: words.metrics[n.metric.kind], value: n.metric.value.toFixed(4) });
  if (!unit && items.length === 0) return null;
  return (
    <div data-run-figures style={{ display: "flex", flexWrap: "wrap", gap: "4px 12px", padding: "2px 10px 8px 10px" }}>
      {unit && (
        <span style={{ ...MONO, fontSize: 11, padding: "1px 8px", borderRadius: 999, border: "1px solid var(--vf-panel-stroke)", color: "var(--vf-text)" }}>
          {t.trainingStory.units[unit.kind](unit.index, unit.total)}
        </span>
      )}
      {items.map((item) => (
        <Figure key={item.label} label={item.label} value={item.value} />
      ))}
    </div>
  );
}

/** The drawing of the old story tab, kept small above the steps when there is
 *  room. Its motion is illustrative, and the caption says so (ADR-118). */
function CompactDrawing({
  story,
  spec,
  tabAccent,
  sheetVisible,
}: {
  story: StoryState;
  spec: StorySpec;
  tabAccent: string;
  sheetVisible: boolean;
}) {
  const s = useT().trainingStory;
  const reducedMotion = usePrefersReducedMotion();
  const documentVisible = useDocumentVisible();
  const theme = useResolvedTheme();
  const { stage, family, phase } = story;
  const familyName = s.families[FAMILY_KEY[family]];
  return (
    <figure
      style={{
        margin: 0,
        padding: "4px 8px 6px",
        background: "rgb(var(--lt-shade, 0 0 0) / 0.45)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 10,
      }}
    >
      <StoryDrawing
        family={family}
        output={spec.output}
        phase={phase}
        accent={accentForStory(spec.task, tabAccent, theme)}
        stage={stage}
        flows={activeFlows(stage, family, phase?.key ?? null)}
        motion={storyMotion({ stage, reducedMotion, documentVisible, sheetVisible })}
        pulseKey={story.epochsSeen}
        label={s.figure(familyName)}
        theme={theme}
      />
      <figcaption style={{ ...MONO, fontSize: 11, lineHeight: 1.45, color: "var(--vf-text-muted)" }}>{s.caption}</figcaption>
    </figure>
  );
}
