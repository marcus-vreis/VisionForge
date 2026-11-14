/**
 * The test setup has no DOM, so each drawing is rendered to markup and read as
 * text. What the browser makes of the animations is the browser check's job; here
 * it is which elements a stage, a motion mode and a family put on the page.
 */
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { StoryFamily, StoryTask } from "../../lib/story-family";
import {
  activeFlows,
  type StoryMotion,
  type StoryPhase,
  type StoryStage,
} from "../../lib/training-story";
import { StoryDrawing } from "./drawings";
import { drawingFor } from "./layouts";
import {
  accentForStory,
  FAILED_COLOR,
  GRADIENT_COLOR,
  IDLE_COLOR,
  INTERRUPTED_COLOR,
  storyPalette,
  toneFor,
  VALIDATION_COLOR,
  withAlpha,
} from "./tone";
import { themeOf } from "./use-viewing";
import type { Theme } from "../../lib/theme";

const FAMILIES: StoryFamily[] = [
  "cnn",
  "transformer",
  "encoder_decoder",
  "detector",
  "autoencoder",
  "patchcore",
];
const ACCENT = "oklch(0.74 0.18 22)";

const phase = (key: StoryPhase["key"], fraction = 0.5): StoryPhase => ({
  key,
  label: key ?? "?",
  done: Math.round(fraction * 10),
  total: 10,
  fraction,
});

/** A drawing, as markup. */
function draw(
  family: StoryFamily,
  stage: StoryStage,
  motion: StoryMotion,
  over: { phase?: StoryPhase | null; accent?: string; output?: "classes" | "value"; theme?: Theme } = {},
) {
  const ph =
    over.phase !== undefined
      ? over.phase
      : family === "patchcore" && stage !== "preparing"
        ? phase("buildingBank")
        : null;
  return renderToStaticMarkup(
    <StoryDrawing
      family={family}
      output={over.output ?? "classes"}
      phase={ph}
      accent={over.accent ?? ACCENT}
      stage={stage}
      flows={activeFlows(stage, family, ph?.key ?? null)}
      motion={motion}
      pulseKey={3}
      label="figure"
      theme={over.theme}
    />,
  );
}

const count = (html: string, needle: string) => html.split(needle).length - 1;

describe("each family's drawing", () => {
  it("is one SVG with its own family mark and a label for a screen reader", () => {
    for (const family of FAMILIES) {
      const html = draw(family, "training", "none");
      expect(html.startsWith("<svg"), family).toBe(true);
      expect(html, family).toContain(`data-family="${family}"`);
      expect(html, family).toContain('role="img"');
      expect(html, family).toContain('aria-label="figure"');
      expect(html, family).toContain("<circle");
      expect(count(html, "<svg"), family).toBe(1);
    }
  });

  it("looks different in every family", () => {
    const shapes = new Set(FAMILIES.map((f) => draw(f, "training", "none")));
    expect(shapes.size).toBe(FAMILIES.length);
  });

  it("draws the detector's three boxes, and only the detector's", () => {
    // Three sheets of data, the thumbnail's frame, its rect object and the boxes.
    const html = draw("detector", "training", "none");
    expect(count(html, 'data-part="box"')).toBe(3);
    expect(count(html, "<rect")).toBe(3 + 1 + 1 + 3);
    for (const family of FAMILIES.filter((f) => f !== "detector" && f !== "patchcore")) {
      // The data source is three small sheets; there is nothing else drawn as a rect.
      expect(count(draw(family, "training", "none"), "<rect"), family).toBe(3);
      expect(draw(family, "training", "none"), family).not.toContain("data-part=");
    }
  });

  it("ends the detector in a thumbnail with the three boxes over its objects, each head linked to it", () => {
    const html = draw("detector", "training", "none");
    expect(count(html, 'data-part="thumbnail"')).toBe(1);
    expect(count(html, 'data-part="thumb-object"')).toBe(3);
    for (const head of [0, 1, 2]) {
      expect(count(html, `data-part="head-link" data-head="${head}"`), `head ${head}`).toBe(1);
      expect(count(html, `data-part="box" data-head="${head}"`), `box ${head}`).toBe(1);
    }
    // The boxes are drawn after the thumbnail, so they sit over it.
    expect(html.indexOf('data-part="thumbnail"')).toBeLessThan(html.indexOf('data-part="box"'));
  });

  it("keeps the boxes inside the thumbnail, small to large", () => {
    const { thumbnail, boxes } = drawingFor("detector", "classes");
    const f = thumbnail!.frame;
    for (const b of boxes) {
      expect(b.x).toBeGreaterThanOrEqual(f.x);
      expect(b.y).toBeGreaterThanOrEqual(f.y);
      expect(b.x + b.w).toBeLessThanOrEqual(f.x + f.w);
      expect(b.y + b.h).toBeLessThanOrEqual(f.y + f.h);
    }
    const areas = boxes.map((b) => b.w * b.h);
    expect(areas).toEqual([...areas].sort((a, b) => a - b));
    expect(drawingFor("cnn", "classes").thumbnail).toBeNull();
  });

  it("sends the detector's forward pulses through the thumbnail's edge to the boxes", () => {
    const d = drawingFor("detector", "classes");
    for (const route of d.forward) {
      expect(route[route.length - 2].x).toBe(d.thumbnail!.frame.x);
    }
  });

  it("draws a regression's network with a one-neuron head", () => {
    const circles = (output: "classes" | "value") =>
      count(draw("cnn", "training", "none", { output }), "<circle");
    expect(circles("classes") - circles("value")).toBe(2);
  });
});

describe("the three flows", () => {
  it("move data, the forward pass, the gradient and validation while training", () => {
    for (const family of FAMILIES.filter((f) => f !== "patchcore")) {
      const html = draw(family, "training", "particles");
      expect(count(html, "<animateMotion"), family).toBeGreaterThan(0);
      // The accent, the amber ring of the gradient, and the cyan diamond of validation.
      expect(html, family).toContain(ACCENT);
      expect(html, family).toContain(GRADIENT_COLOR);
      expect(html, family).toContain(VALIDATION_COLOR);
      // The validation pulse waits to be fired by an epoch.
      expect(html, family).toContain('begin="indefinite"');
    }
  });

  it("gives each flow a shape as well as a colour", () => {
    const html = draw("cnn", "training", "particles");
    // Ring: unfilled circles. Diamond: rotated squares.
    expect(html).toContain('fill="none" stroke="' + GRADIENT_COLOR);
    expect(html).toContain('transform="rotate(45)"');
  });

  it("brings only data in while preparing: no gradient and no validation", () => {
    const html = draw("cnn", "preparing", "particles");
    expect(count(html, "<animateMotion")).toBeGreaterThan(0);
    expect(html).not.toContain(GRADIENT_COLOR);
    expect(html).not.toContain(VALIDATION_COLOR);
  });

  it("runs only validation while testing", () => {
    const html = draw("cnn", "testing", "particles");
    expect(html).toContain(VALIDATION_COLOR);
    expect(html).not.toContain(GRADIENT_COLOR);
  });

  it("keeps the number of animated elements small", () => {
    for (const family of FAMILIES) {
      for (const stage of ["preparing", "training", "testing"] as const) {
        const html = draw(family, stage, "particles");
        expect(count(html, "<animateMotion"), `${family} ${stage}`).toBeLessThanOrEqual(30);
      }
    }
  });

  it("draws nothing moving when the run is not going or nobody looks", () => {
    for (const family of FAMILIES) {
      for (const stage of ["queued", "done", "interrupted", "failed"] as const) {
        const html = draw(family, stage, "none");
        expect(html, `${family} ${stage}`).not.toContain("animateMotion");
        expect(html, `${family} ${stage}`).not.toContain("<animate ");
      }
      expect(draw(family, "training", "none"), family).not.toContain("animateMotion");
    }
  });

  it("draws arrows and no animation for a reader who asked for less motion", () => {
    for (const family of FAMILIES) {
      for (const stage of ["preparing", "training", "testing"] as const) {
        const html = draw(family, stage, "arrows");
        expect(html, `${family} ${stage}`).not.toContain("animateMotion");
        expect(html, `${family} ${stage}`).not.toContain("<animate ");
        // A chevron: a small open path placed and turned along a curve.
        expect(html, `${family} ${stage}`).toContain("rotate(");
        expect(html, `${family} ${stage}`).toContain("M-4.2 -3.6L1.8 0L-4.2 3.6");
      }
    }
  });

  it("points the gradient's arrows backwards", () => {
    const html = draw("cnn", "training", "arrows");
    const group = new RegExp(`stroke="${GRADIENT_COLOR.replace(/[().]/g, "\\$&")}"[^>]*>(.*?)</g>`).exec(html);
    expect(group).not.toBeNull();
    const amber = group![1];
    const angles = [...amber.matchAll(/rotate\((-?[\d.]+)\)/g)].map((m) => Math.abs(Number(m[1])));
    expect(angles.length).toBeGreaterThan(0);
    // Pointing left: a heading well past a quarter turn.
    expect(angles.every((a) => a > 90)).toBe(true);
  });
});

describe("PatchCore's phases", () => {
  it("lights the memory bank with the progress of its phase", () => {
    const lit = (fraction: number) => {
      const html = draw("patchcore", "training", "none", { phase: phase("buildingBank", fraction) });
      return count(html, 'fill-opacity="0.85"');
    };
    expect(lit(0)).toBe(0);
    expect(lit(0.5)).toBe(15);
    expect(lit(1)).toBe(30);
  });

  it("scores with the validation colour, and stores with the accent", () => {
    const scoring = draw("patchcore", "training", "particles", { phase: phase("scoring") });
    expect(scoring).toContain(VALIDATION_COLOR);
    const storing = draw("patchcore", "training", "particles", { phase: phase("buildingBank") });
    expect(storing).not.toContain(`fill="${VALIDATION_COLOR}" fill-opacity="0.5"`);
    expect(storing).not.toContain(GRADIENT_COLOR);
  });

  it("has no gradient in any phase", () => {
    for (const key of ["extractingFeatures", "buildingBank", "scoring"] as const) {
      expect(draw("patchcore", "training", "particles", { phase: phase(key) })).not.toContain(
        GRADIENT_COLOR,
      );
    }
  });

  it("highlights the zone of the phase it is in", () => {
    const zoneOpacity = (key: StoryPhase["key"]) =>
      [...draw("patchcore", "training", "none", { phase: phase(key) }).matchAll(/stroke-opacity="0\.5"/g)].length;
    expect(zoneOpacity("extractingFeatures")).toBeGreaterThanOrEqual(1);
  });
});

describe("the colours of a drawing", () => {
  it("goes by the task's own accent", () => {
    expect(accentForStory("classification", undefined)).toBe("oklch(0.74 0.18 22)");
    expect(accentForStory("detection", undefined)).toBe("oklch(0.78 0.18 150)");
    expect(accentForStory("regression", undefined)).toBe("oklch(0.74 0.16 240)");
    expect(accentForStory("segmentation", undefined)).toBe("oklch(0.74 0.18 305)");
    expect(accentForStory("anomaly", undefined)).toBe("oklch(0.80 0.15 75)");
  });

  it("gives a researcher's own task the accent of its tab, unless that is a variable", () => {
    expect(accentForStory("custom", "oklch(0.7 0.1 10)")).toBe("oklch(0.7 0.1 10)");
    const neutral = accentForStory("custom", "var(--vf-text-muted)");
    expect(neutral.startsWith("oklch(")).toBe(true);
    expect(accentForStory("custom", undefined)).toBe(neutral);
  });

  it("never hands the drawing a CSS variable", () => {
    const tasks: StoryTask[] = ["classification", "detection", "regression", "segmentation", "anomaly", "custom"];
    for (const task of tasks) {
      for (const stage of ["queued", "preparing", "training", "testing", "done", "interrupted", "failed"] as const) {
        const tone = toneFor(stage, accentForStory(task, "var(--x)"));
        expect(tone.line, `${task} ${stage}`).not.toContain("var(");
        expect(tone.web, `${task} ${stage}`).not.toContain("var(");
      }
    }
  });

  it("tells the end states apart: amber for a stop, red for a failure, filled when done", () => {
    expect(toneFor("interrupted", ACCENT).line).toBe(INTERRUPTED_COLOR);
    expect(toneFor("failed", ACCENT).line).toBe(FAILED_COLOR);
    expect(toneFor("done", ACCENT).fillOpacity).toBeGreaterThan(toneFor("training", ACCENT).fillOpacity);
    expect(toneFor("preparing", ACCENT).line).not.toBe(ACCENT);
  });

  it("adds an opacity to an oklch colour, and leaves the rest alone", () => {
    expect(withAlpha("oklch(0.74 0.18 22)", 0.4)).toBe("oklch(0.74 0.18 22 / 0.4)");
    expect(withAlpha("rgba(1, 2, 3, 0.9)", 0.4)).toBe("rgba(1, 2, 3, 0.9)");
  });

  const TASKS: StoryTask[] = ["classification", "detection", "regression", "segmentation", "anomaly", "custom"];
  const STAGES = ["queued", "preparing", "training", "testing", "done", "interrupted", "failed"] as const;
  /** The lightness of an `oklch(L C H)` literal. */
  const lightness = (color: string) => Number(/^oklch\(([\d.]+) /.exec(color)?.[1]);

  it("keeps the dark theme's colours as they were", () => {
    expect(storyPalette()).toEqual(storyPalette("dark"));
    expect(storyPalette("dark")).toMatchObject({
      failed: FAILED_COLOR,
      interrupted: INTERRUPTED_COLOR,
      gradient: GRADIENT_COLOR,
      validation: VALIDATION_COLOR,
      idle: IDLE_COLOR,
    });
    for (const task of TASKS) {
      expect(accentForStory(task, undefined, "dark")).toBe(accentForStory(task, undefined));
      for (const stage of STAGES) {
        expect(toneFor(stage, ACCENT, "dark")).toEqual(toneFor(stage, ACCENT));
      }
    }
  });

  it("darkens every tone in the light theme, keeping its hue, and never hands over a variable", () => {
    const dark = storyPalette("dark");
    const light = storyPalette("light");
    for (const key of ["failed", "interrupted", "gradient", "validation", "custom"] as const) {
      expect(lightness(light[key]), key).toBeLessThanOrEqual(0.56);
      // Lightness mirrored around 0.65, chroma and hue untouched.
      expect(lightness(light[key]), key).toBeCloseTo(1.3 - lightness(dark[key]), 5);
      expect(light[key].split(" ").slice(1).join(" "), key).toBe(dark[key].split(" ").slice(1).join(" "));
    }
    expect(light.idle).not.toBe(IDLE_COLOR);
    expect(light.idle.startsWith("rgba(60, 66, 80")).toBe(true);
    for (const task of TASKS) {
      const accent = accentForStory(task, "var(--x)", "light");
      expect(lightness(accent), task).toBeLessThanOrEqual(0.56);
      for (const stage of STAGES) {
        const tone = toneFor(stage, accent, "light");
        expect(tone.line, `${task} ${stage}`).not.toContain("var(");
        expect(tone.web, `${task} ${stage}`).not.toContain("var(");
      }
    }
  });

  it("matches the light theme's task accents to the tabs' (index.css)", () => {
    expect(accentForStory("classification", undefined, "light")).toBe("oklch(0.56 0.18 22)");
    expect(accentForStory("detection", undefined, "light")).toBe("oklch(0.52 0.15 150)");
    expect(accentForStory("regression", undefined, "light")).toBe("oklch(0.52 0.16 240)");
    expect(accentForStory("segmentation", undefined, "light")).toBe("oklch(0.52 0.18 305)");
    expect(accentForStory("anomaly", undefined, "light")).toBe("oklch(0.50 0.15 75)");
    expect(accentForStory("custom", "oklch(0.7 0.1 10)", "light")).toBe("oklch(0.7 0.1 10)");
  });

  it("tells the end states apart in the light theme too", () => {
    const light = storyPalette("light");
    expect(toneFor("interrupted", ACCENT, "light").line).toBe(light.interrupted);
    expect(toneFor("failed", ACCENT, "light").line).toBe(light.failed);
    expect(toneFor("queued", ACCENT, "light").line).toBe(light.idle);
  });

  it("draws in the light colours when given the light theme", () => {
    const light = storyPalette("light");
    const waiting = draw("cnn", "queued", "none", { theme: "light" });
    expect(waiting).toContain(light.idle);
    expect(waiting).not.toContain(IDLE_COLOR);
    const learning = draw("cnn", "training", "particles", { theme: "light" });
    expect(learning).toContain(light.gradient);
    expect(learning).toContain(light.validation);
    expect(learning).not.toContain(GRADIENT_COLOR);
    expect(learning).not.toContain(VALIDATION_COLOR);
    const scoring = draw("patchcore", "testing", "particles", { theme: "light", phase: phase("scoring") });
    expect(scoring).toContain(light.validation);
    expect(scoring).not.toContain(VALIDATION_COLOR);
  });

  it("reads the page's theme from data-theme, dark unless it says light", () => {
    expect(themeOf("light")).toBe("light");
    expect(themeOf("dark")).toBe("dark");
    expect(themeOf(undefined)).toBe("dark");
    expect(themeOf("sepia")).toBe("dark");
  });

  it("paints a drawing in the accent it is given", () => {
    for (const task of ["classification", "detection", "regression", "segmentation"] as const) {
      const accent = accentForStory(task, undefined);
      expect(draw("cnn", "training", "none", { accent })).toContain(accent);
    }
  });
});
