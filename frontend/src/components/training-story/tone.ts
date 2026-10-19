/** The colours of a drawing in each stage (ADR-118). Pure, so the choices are
 * covered by vitest. Every colour is a literal (`oklch(...)`, `rgba(...)`) and
 * never a CSS variable: the drawings set them as SVG attributes, where a variable
 * is not guaranteed to resolve. The light theme (ADR-125) gets its own literals,
 * picked by the resolved theme the drawing is given.
 */

import { TASK_ACCENT, TASK_ACCENT_LIGHT } from "../../lib/task-accent";
import type { StoryTask } from "../../lib/story-family";
import type { Theme } from "../../lib/theme";
import type { StoryStage } from "../../lib/training-story";

/** The red of a failure, as the rest of the sheet words it. */
export const FAILED_COLOR = "oklch(0.74 0.18 22)";
/** The amber of a run a stop cut (lib/unit-status.ts `STOPPED_COLOR`). */
export const INTERRUPTED_COLOR = "oklch(0.84 0.12 85)";
/** The gradient coming back. Close to the anomaly accent (also an amber), which
 *  is why every flow also has a shape of its own: dot, ring, diamond. */
export const GRADIENT_COLOR = "oklch(0.82 0.16 65)";
/** The validation pass. */
export const VALIDATION_COLOR = "oklch(0.86 0.13 190)";
/** What has not started, or is waiting. */
export const IDLE_COLOR = "rgba(231, 233, 238, 0.9)";
/** The accent of a researcher's own task, which has none of its own. */
const CUSTOM_ACCENT = "oklch(0.78 0.10 250)";

/** The fixed colours of a drawing, apart from the task's accent. */
export interface StoryPalette {
  failed: string;
  interrupted: string;
  gradient: string;
  validation: string;
  idle: string;
  custom: string;
}

const DARK_PALETTE: StoryPalette = {
  failed: FAILED_COLOR,
  interrupted: INTERRUPTED_COLOR,
  gradient: GRADIENT_COLOR,
  validation: VALIDATION_COLOR,
  idle: IDLE_COLOR,
  custom: CUSTOM_ACCENT,
};

/** The dark tones with their lightness mirrored around 0.65 (0.86 → 0.44), as
 *  ADR-125 does for the sheet's text; idle is index.css's `--lt-idle`. */
const LIGHT_PALETTE: StoryPalette = {
  failed: "oklch(0.56 0.18 22)",
  interrupted: "oklch(0.46 0.12 85)",
  gradient: "oklch(0.48 0.16 65)",
  validation: "oklch(0.44 0.13 190)",
  idle: "rgba(60, 66, 80, 0.85)",
  custom: "oklch(0.52 0.10 250)",
};

/** The colours a drawing goes by in a theme. The dark theme keeps the originals. */
export function storyPalette(theme: Theme = "dark"): StoryPalette {
  return theme === "light" ? LIGHT_PALETTE : DARK_PALETTE;
}

/** The accent a task's drawing goes by: the task's own colour (the same one the
 *  history and the run's detail use), or, for a researcher's own task, the accent
 *  of the tab it was started from when that is a literal colour. */
export function accentForStory(task: StoryTask, tabAccent: string | undefined, theme: Theme = "dark"): string {
  if (task !== "custom") return (theme === "light" ? TASK_ACCENT_LIGHT : TASK_ACCENT)[task];
  return tabAccent && !tabAccent.startsWith("var(") ? tabAccent : storyPalette(theme).custom;
}

/** `oklch(L C H)` with an opacity: `oklch(L C H / a)`. Any other colour is
 *  returned as it came, since only the accents go through here. */
export function withAlpha(color: string, alpha: number): string {
  const m = /^oklch\(([^)/]+)\)$/.exec(color);
  return m ? `oklch(${m[1].trim()} / ${alpha})` : color;
}

export interface Tone {
  /** Outline of a neuron and of what belongs to the family (boxes, arcs). */
  line: string;
  lineOpacity: number;
  /** How much of the colour fills a neuron. */
  fillOpacity: number;
  /** The faint web between the columns. */
  web: string;
  webOpacity: number;
}

/** How the network looks in a stage: dim while it waits and prepares, in the
 *  accent while it learns, filled when it is done, amber when a stop cut it, red
 *  when it failed. */
export function toneFor(stage: StoryStage, accent: string, theme: Theme = "dark"): Tone {
  const { idle, interrupted, failed } = storyPalette(theme);
  switch (stage) {
    case "queued":
    case "preparing":
      return { line: idle, lineOpacity: 0.34, fillOpacity: 0.04, web: idle, webOpacity: 0.09 };
    case "training":
    case "testing":
      return { line: accent, lineOpacity: 0.8, fillOpacity: 0.1, web: accent, webOpacity: 0.2 };
    case "done":
      return { line: accent, lineOpacity: 0.9, fillOpacity: 0.3, web: accent, webOpacity: 0.26 };
    case "interrupted":
      return { line: interrupted, lineOpacity: 0.7, fillOpacity: 0.14, web: interrupted, webOpacity: 0.16 };
    case "failed":
      return { line: failed, lineOpacity: 0.55, fillOpacity: 0.06, web: failed, webOpacity: 0.12 };
  }
}
