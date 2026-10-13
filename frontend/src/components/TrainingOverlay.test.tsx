/**
 * The tabs of the training sheet (ADR-118), rendered to markup: the setup has no
 * DOM, so what is read is which tab is open and what each one puts on the page.
 * Clicking is the browser check's.
 */
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { storySpecFor } from "../lib/story-family";
import type { SheetTab } from "../lib/training-story";
import type { RunStatus, TrainingEvent } from "../types/run";
import { TrainingOverlay } from "./TrainingOverlay";

const running: RunStatus = { status: "running", run_id: "run-1", error: null };
const events: TrainingEvent[] = [
  { event: "start", total_epochs: 4 },
  {
    event: "epoch_end",
    epoch: 1,
    total_epochs: 4,
    train_loss: 0.7,
    val_loss: 0.8,
    val_accuracy: 0.5,
  },
];

function sheet(lang: "pt" | "en", tab: SheetTab, visible = true) {
  return renderToStaticMarkup(
    <LanguageContext.Provider
      value={{
        lang,
        t: lang === "pt" ? pt : en,
        locale: lang === "pt" ? "pt-BR" : "en-US",
        setLang: () => {},
      }}
    >
      <TrainingOverlay
        status={running}
        progressEvents={events}
        visible={visible}
        taskAccent="oklch(0.74 0.18 22)"
        taskLabel="Classificação"
        taskKey="classification"
        story={storySpecFor("classification", { model: { name: "vit_b_16" } })}
        tab={tab}
        onTabChange={() => {}}
        onClose={() => {}}
      />
    </LanguageContext.Provider>,
  );
}

describe("the training sheet's tabs", () => {
  it("offers the log and 'Como funciona', side by side", () => {
    const html = sheet("pt", "log");
    expect(html).toContain('role="tablist"');
    const log = html.indexOf(`>${pt.trainingStory.logTab}<`);
    const story = html.indexOf(`>${pt.trainingStory.tab}<`);
    expect(log).toBeGreaterThan(-1);
    expect(story).toBeGreaterThan(log);
    expect(pt.trainingStory.tab).toBe("Como funciona");
  });

  it("opens on the log, with the story left out until it is chosen", () => {
    const html = sheet("pt", "log");
    expect(html).toMatch(/id="training-tab-log"[^>]*aria-selected="true"/s);
    expect(html).toMatch(/id="training-tab-story"[^>]*aria-selected="false"/s);
    expect(html).not.toContain("data-story-stage");
    expect(html).not.toContain("animateMotion");
    // The log is in the page and in view.
    expect(html).toContain("$ visionforge train --task classification");
    expect(html).not.toMatch(/id="training-panel-log"[^>]*hidden/);
  });

  it("shows the story when it is the open tab, and keeps the log in the page, hidden", () => {
    const html = sheet("en", "story");
    expect(html).toMatch(/id="training-tab-story"[^>]*aria-selected="true"/s);
    expect(html).toContain('data-story-stage="training"');
    expect(html).toContain(en.trainingStory.stages.training);
    // The run's own model, not the tab's: a ViT is a transformer.
    expect(html).toContain('data-family="transformer"');
    expect(html).toMatch(/id="training-panel-log"[^>]*hidden/s);
    expect(html).toContain("$ visionforge train --task classification");
  });

  it("lets a minimized sheet's drawing stand still", () => {
    expect(sheet("pt", "story", true)).toContain("animateMotion");
    expect(sheet("pt", "story", false)).not.toContain("animateMotion");
  });
});
