/**
 * The "Resultados" tab, rendered to markup: which card, charts and table the
 * events put on the page. The drawing itself is the browser check's.
 */
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { storySpecFor, type StorySpec } from "../lib/story-family";
import type { TrainingEvent } from "../types/run";
import { LiveResults } from "./LiveResults";

const classification = storySpecFor("classification", { model: { name: "resnet18" } });

function render(events: TrainingEvent[], spec: StorySpec = classification, lang: "pt" | "en" = "pt") {
  return renderToStaticMarkup(
    <LanguageContext.Provider
      value={{
        lang,
        t: lang === "pt" ? pt : en,
        locale: lang === "pt" ? "pt-BR" : "en-US",
        setLang: () => {},
      }}
    >
      <LiveResults events={events} spec={spec} accent="oklch(0.74 0.18 22)" />
    </LanguageContext.Provider>,
  );
}

function epoch(n: number, valLoss: number, extra: Record<string, unknown> = {}): TrainingEvent {
  return {
    event: "epoch_end",
    epoch: n,
    total_epochs: 3,
    train_loss: 1 / n,
    val_loss: valLoss,
    val_accuracy: 0.5 + n / 10,
    ...extra,
  } as TrainingEvent;
}

describe("LiveResults", () => {
  it("waits for the first epoch before drawing anything", () => {
    const html = render([{ event: "start", total_epochs: 3 }]);
    expect(html).toContain(pt.liveResults.empty);
    expect(html).not.toContain("<svg");
  });

  it("says PatchCore has no epochs instead of waiting for one", () => {
    const spec = storySpecFor("anomaly", { model: { name: "patchcore" } });
    const html = render([{ event: "phase", label: "memory_bank", done: 1, total: 4 }], spec);
    expect(html).toContain(pt.liveResults.noEpochsPatchcore);
  });

  it("names the epoch of the saved checkpoint and marks it on the charts", () => {
    const html = render([epoch(1, 0.9), epoch(2, 0.3), epoch(3, 0.5)], classification, "en");
    expect(html).toContain(en.liveResults.best(2, en.compareRuns.curves.val_loss, "0.3000"));
    expect(html).toContain(en.compareRuns.direction.lower);
    // One marker per chart: the loss and the headline metric.
    expect(html.match(/data-best-epoch="2"/g)).toHaveLength(2);
    expect(html).toContain(`aria-label="${en.compareRuns.curves.val_accuracy}"`);
  });

  it("draws the curves of the trial in flight and lists the closed ones", () => {
    const html = render([
      { event: "trial_start", trial_index: 0, total_trials: 2, overrides: { lr: 0.1 }, seed: 7 },
      epoch(1, 0.8),
      epoch(2, 0.6),
      { event: "trial_end", trial_index: 0, total_trials: 2, total_epochs: 2 },
      { event: "trial_start", trial_index: 1, total_trials: 2, overrides: { lr: 0.01 }, seed: 7 },
      epoch(1, 0.7),
    ]);
    expect(html).toContain(pt.liveResults.unitsTitle);
    expect(html).toContain("lr=0.1");
    expect(html).toContain(pt.liveResults.bestShort(2, pt.compareRuns.curves.val_loss, "0.6000"));
    // The card is the running trial's, not the sweep's best.
    expect(html).toContain(pt.liveResults.best(1, pt.compareRuns.curves.val_loss, "0.7000"));
    expect(html).not.toContain("lr=0.01");
  });

  it("draws a number that never moved as one level line with one label", () => {
    const flat = [1, 2, 3].map((n) => ({ ...epoch(n, 0.5 - n / 10), val_accuracy: 1 })) as TrainingEvent[];
    const html = render(flat);
    const accuracy = html.slice(html.indexOf(`aria-label="${pt.compareRuns.curves.val_accuracy}"`));
    expect(accuracy.match(/>1\.0000</g)).toHaveLength(1);
  });

  it("states no best epoch for a researcher's own task", () => {
    const spec = storySpecFor("custom:my_task", {});
    const html = render([epoch(1, 0.4), epoch(2, 0.2)], spec);
    expect(html).not.toContain(pt.liveResults.bestNote);
    expect(html).toContain("<svg");
  });

  it("warns of overfitting from the reported losses", () => {
    const events = [
      { event: "epoch_end", epoch: 1, total_epochs: 5, train_loss: 0.9, val_loss: 0.5, val_accuracy: 0.6 },
      { event: "epoch_end", epoch: 2, total_epochs: 5, train_loss: 0.7, val_loss: 0.6, val_accuracy: 0.6 },
      { event: "epoch_end", epoch: 3, total_epochs: 5, train_loss: 0.5, val_loss: 0.7, val_accuracy: 0.6 },
      { event: "epoch_end", epoch: 4, total_epochs: 5, train_loss: 0.3, val_loss: 0.8, val_accuracy: 0.6 },
    ] as TrainingEvent[];
    expect(render(events)).toContain(pt.trainingStory.overfitting(2));
  });
});
