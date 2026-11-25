/**
 * "Como funciona": the run as steps and the real code of the step it is in. The
 * setup has no DOM, so the tab is rendered to markup and read as text; clicking
 * is covered by the follow/pause reducer in lib/run-steps.test.ts.
 */
import type { ComponentProps } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { ApiError, type CodeNote, type CodeSection } from "../../api/client";
import { LanguageContext } from "../../i18n/context";
import { en } from "../../i18n/en";
import { pt } from "../../i18n/pt";
import { lineMarks } from "../../lib/code-lines";
import { libraryNotice, tourStatus, tourSupport, unavailableReason } from "../../lib/code-tour";
import { RUN_STEPS, type RunStep } from "../../lib/run-steps";
import { storySpecFor } from "../../lib/story-family";
import { SHEET_TABS } from "../../lib/training-story";
import type { TrainingEvent } from "../../types/run";
import { RunCodePanel } from "./TrainingCodeTab";

const count = (html: string, needle: string) => html.split(needle).length - 1;

const inLanguage = (lang: "pt" | "en", node: React.ReactNode) =>
  renderToStaticMarkup(
    <LanguageContext.Provider
      value={{ lang, t: lang === "pt" ? pt : en, locale: lang === "pt" ? "pt-BR" : "en-US", setLang: () => {} }}
    >
      {node}
    </LanguageContext.Provider>,
  );

const note = (first: number, last: number, active: boolean | null, text = `note ${first}`): CodeNote => ({
  first,
  last,
  text,
  active,
});

const section = (id: string, step: RunStep, over: Partial<CodeSection> = {}): CodeSection => ({
  id,
  step,
  summary: `What ${id} does.`,
  name: `Trainer._${id}`,
  file: "visionforge/core/trainer.py",
  first_line: 100,
  lines: [`def _${id}(self):`, "    model.train()", "    return 1"],
  notes: [],
  notes_resolved: 0,
  notes_total: 0,
  ...over,
});

const TOUR: CodeSection[] = RUN_STEPS.map((step) =>
  step === "train"
    ? section("train", "train", {
        lines: ["def _train_epoch(self):", "    model.train()", "    if use_amp:", "        scaler.step()", "    else:", "        optimizer.step()"],
        notes: [note(1, 1, null, "Training mode."), note(2, 3, false, "Mixed precision."), note(4, 5, true, "The plain step.")],
        notes_resolved: 3,
        notes_total: 3,
      })
    : section(step, step),
);

const start: TrainingEvent = { event: "start", total_epochs: 4 };
const epoch = (n: number, total = 4): TrainingEvent => ({
  event: "epoch_end",
  epoch: n,
  total_epochs: total,
  train_loss: 0.5,
  val_loss: 0.6,
  val_accuracy: 0.8,
});
const batch: TrainingEvent = { event: "batch", epoch: 3, batch: 5, total_batches: 10, total_epochs: 4 };

const tab = (lang: "pt" | "en", over: Partial<ComponentProps<typeof RunCodePanel>> = {}) =>
  inLanguage(
    lang,
    <RunCodePanel
      events={[start, epoch(2)]}
      status="running"
      stopped={false}
      unitKind={null}
      spec={storySpecFor("classification", { model: { name: "resnet18" } })}
      accent="oklch(0.74 0.18 22)"
      sheetVisible
      tour={{ kind: "ready", sections: TOUR }}
      {...over}
    />,
  );

describe("the training sheet's tabs", () => {
  it("are the log, the results and 'Como funciona', where the code now lives", () => {
    expect(SHEET_TABS).toEqual(["log", "results", "story"]);
  });
});

describe("the steps", () => {
  it("lists the run's steps in order, each with its sentence", () => {
    const html = tab("pt");
    let at = -1;
    for (const step of RUN_STEPS) {
      const next = html.indexOf(`data-step="${step}"`);
      expect(next).toBeGreaterThan(at);
      at = next;
      expect(html).toContain(pt.codeTour.steps[step].text);
    }
  });

  it("marks the group of training and validation while the epoch has no batch report", () => {
    const html = tab("pt");
    expect(count(html, 'aria-current="step"')).toBe(2);
    expect(html).toMatch(/data-step="train" data-running="true"/);
    expect(html).toMatch(/data-step="validate" data-running="true"/);
    expect(html).toContain(pt.codeTour.liveGroup(["época de treino", "validação"]));
    expect(html).not.toContain(`>${pt.codeTour.live}<`);
  });

  it("marks only the training once a batch is reported, and says the code is running", () => {
    const html = tab("en", { events: [start, epoch(2), batch] });
    expect(count(html, 'aria-current="step"')).toBe(1);
    expect(html).toMatch(/data-step="train" data-running="true"/);
    expect(html).toContain(en.codeTour.live);
    expect(html).toContain('data-run-step="train"');
  });

  it("marks data, model and loss together before the trainer starts", () => {
    const html = tab("pt", { events: [] });
    expect(count(html, 'aria-current="step"')).toBe(3);
    expect(html).toContain('data-run-step="data"');
    expect(html).toContain(pt.codeTour.liveGroup(["dados", "modelo", "loss"]));
  });

  it("marks nothing as running once the run is over, and rests on the test", () => {
    const events = [start, epoch(4), { event: "end", total_epochs: 4 } as TrainingEvent];
    const html = tab("pt", { events, status: "completed" });
    expect(html).toContain('data-story-stage="done"');
    expect(html).not.toContain('aria-current="step"');
    expect(html).not.toContain("data-live");
    expect(html).not.toContain("data-resume");
    expect(html).toContain('data-run-step="test"');
  });

  it("lists only the steps the tour has code for", () => {
    const tour: CodeSection[] = [
      section("yolo_args", "train"),
      section("yolo_augment", "data"),
      section("yolo_fit", "model"),
      section("yolo_epoch", "validate"),
      section("yolo_loop", "train"),
    ];
    const html = tab("pt", { tour: { kind: "ready", sections: tour } });
    expect(html).not.toContain('data-step="loss"');
    expect(html).not.toContain('data-step="test"');
    // Ultralytics' own loop is the code of an epoch in flight, not the arguments.
    expect(html).toContain('data-code-view="yolo_loop"');
  });

  it("words PatchCore's training and validation as what it does", () => {
    const spec = storySpecFor("anomaly", { model: { name: "patchcore" } });
    const events: TrainingEvent[] = [{ event: "start", total_epochs: 1 }, { event: "phase", label: "montando o banco", done: 40, total: 100 }];
    const html = tab("pt", { spec, events });
    expect(html).toContain(pt.codeTour.patchcoreSteps.train.title);
    expect(html).toContain(pt.codeTour.patchcoreSteps.validate.text);
    expect(html).not.toContain(pt.codeTour.steps.train.title);
  });

  it("has a title and a sentence for every step, in both languages", () => {
    for (const step of RUN_STEPS) {
      for (const dict of [pt, en]) {
        expect(dict.codeTour.steps[step].title).toBeTruthy();
        expect(dict.codeTour.steps[step].text).toBeTruthy();
      }
      expect(pt.codeTour.steps[step].text).not.toBe(en.codeTour.steps[step].text);
    }
  });

  it("keeps every sentence of the tab short: two lines of its column at most", () => {
    for (const dict of [pt, en]) {
      const sentences = [...Object.values(dict.codeTour.steps), ...Object.values(dict.codeTour.patchcoreSteps)].map((s) => s.text);
      for (const text of [...sentences, dict.codeTour.granularity]) expect(text.length).toBeLessThanOrEqual(90);
    }
  });
});

describe("the figures on the running step", () => {
  it("shows the epoch, the losses and the headline metric the trainer reported", () => {
    const html = tab("pt");
    expect(html).toContain("data-run-figures");
    expect(html).toContain(pt.trainingStory.numbers.epochOf(2, 4));
    expect(html).toContain("0.5000");
    expect(html).toContain("0.6000");
    expect(html).toContain("0.8000");
    expect(html).toContain(pt.trainingStory.numbers.metrics.accuracy);
    // Under the last step of the running group, after validation's sentence.
    expect(html.indexOf("data-run-figures")).toBeGreaterThan(html.indexOf(pt.codeTour.steps.validate.text));
  });

  it("does not show a number the trainer has not reported", () => {
    const html = tab("en", { events: [start] });
    expect(html).not.toContain(en.trainingStory.numbers.trainLoss);
    expect(html).not.toContain(en.trainingStory.numbers.metrics.accuracy);
    expect(html).toContain(en.trainingStory.numbers.epochOf(0, 4));
  });

  it("names the unit of a multi-unit job", () => {
    const events: TrainingEvent[] = [
      { event: "trial_start", trial_index: 1, total_trials: 5, overrides: {}, seed: 0 },
      start,
      epoch(1),
    ];
    expect(tab("pt", { events, unitKind: "fold" })).toContain("dobra 2/5");
    expect(tab("en", { events, unitKind: "trial" })).toContain("trial 2/5");
    expect(tab("en")).not.toContain("trial ");
  });

  it("shows PatchCore's phase and its progress where the others show an epoch", () => {
    const spec = storySpecFor("anomaly", { model: { name: "patchcore" } });
    const events: TrainingEvent[] = [{ event: "start", total_epochs: 1 }, { event: "phase", label: "montando o banco", done: 40, total: 100 }];
    const html = tab("pt", { spec, events });
    expect(html).toContain(`${pt.trainingOverlay.phases.buildingBank} · 40/100`);
    expect(html).not.toContain(pt.trainingStory.numbers.epoch + "<");
    expect(html).not.toContain(pt.trainingStory.numbers.trainLoss);
  });
});

describe("the code of the step", () => {
  const trainHtml = () => tab("pt", { events: [start, epoch(2), batch] });

  it("shows the function's real lines, numbered as in the file, with its sentence", () => {
    const html = trainHtml();
    expect(html).toContain('data-code-view="train"');
    expect(html).toContain("def _train_epoch(self):");
    expect(html).toContain(">100<");
    expect(html).toContain(">105<");
    expect(html).toContain("What train does.");
    expect(html).toContain(pt.codeTour.where("visionforge/core/trainer.py", "Trainer._train", 100));
  });

  it("keeps the notes closed, with a quiet marker per noted range", () => {
    const html = trainHtml();
    expect(count(html, "data-note-marker")).toBe(3);
    expect(count(html, 'aria-expanded="false"')).toBe(3);
    expect(html).toContain(`aria-label="${pt.codeTour.noteToggle(pt.codeTour.lines(102, 103))}"`);
    expect(html).not.toContain("Mixed precision.");
    expect(html).not.toContain('role="note"');
  });

  it("fades the lines of a branch this run does not take", () => {
    const marks = lineMarks(TOUR[3].notes, TOUR[3].lines.length);
    expect(marks.map((m) => m.faded)).toEqual([false, false, true, true, false, false]);
    expect(marks.map((m) => m.noted)).toEqual([false, true, true, true, true, true]);
    expect(marks[2].starts).toEqual([1]);
    expect(marks[3].ends).toEqual([1]);
    expect(count(trainHtml(), "opacity:0.45")).toBe(2);
  });

  it("never fades a line this run's path covers, even inside a skipped branch", () => {
    const marks = lineMarks([note(0, 3, false), note(1, 1, true)], 4);
    expect(marks.map((m) => m.faded)).toEqual([true, false, true, true]);
  });

  it("lists the functions of a step that has several, with name and file", () => {
    // Detection's data step: the dataset, its labels and the batch.
    const tour: CodeSection[] = [
      section("data", "data", { name: "DetectionDataset.__getitem__", file: "visionforge/core/detection_dataset.py" }),
      section("labels", "data", { name: "DetectionDataset._read_labels", file: "visionforge/core/detection_dataset.py" }),
      ...TOUR.filter((s) => s.step !== "data"),
    ];
    // Before the trainer starts, the data step is the first of the group shown.
    const html = tab("en", { tour: { kind: "ready", sections: tour }, events: [] });
    expect(html).toContain(`aria-label="${en.codeTour.functionsLabel}"`);
    expect(html).toContain("DetectionDataset._read_labels");
    expect(html).toContain("· visionforge/core/detection_dataset.py");
    expect(count(html, 'aria-pressed="true"')).toBe(2); // the step and its first function
    // A one-function step has no list.
    expect(tab("en")).not.toContain(`aria-label="${en.codeTour.functionsLabel}"`);
  });

  it("names a library's code with its version, license and resolved notes", () => {
    const loop = section("yolo_loop", "train", {
      name: "BaseTrainer._do_train",
      file: "ultralytics/engine/trainer.py",
      notes_resolved: 9,
      notes_total: 10,
      library: { name: "ultralytics", version: "8.4.60", license: "AGPL-3.0" },
    });
    const notice = libraryNotice(loop, pt.codeTour);
    expect(notice).toContain("ultralytics 8.4.60");
    expect(notice).toContain("AGPL-3.0");
    expect(notice).toContain("9 de 10");
    expect(libraryNotice({ ...loop, library: null }, pt.codeTour)).toBeNull();
    expect(libraryNotice(loop, en.codeTour)).toContain("9 of 10");
    const html = tab("pt", { tour: { kind: "ready", sections: [loop] }, events: [start, batch] });
    expect(html).toContain(notice ?? "-");
  });

  it("says the code is loading, missing or unavailable instead of failing", () => {
    expect(tab("pt", { tour: { kind: "loading" } })).toContain(pt.codeTour.loading);
    expect(tab("pt", { tour: { kind: "missing" } })).toContain(pt.codeTour.missing);
    expect(tab("pt", { tour: { kind: "unavailable", reason: "Sem tour." } })).toContain("Sem tour.");
    expect(tab("pt", { tour: { kind: "failed" } })).toContain(pt.codeTour.failed);
    // Without a tour the run is still the six steps.
    expect(count(tab("pt", { tour: { kind: "missing" } }), "data-step=")).toBe(6);
  });

  it("decides support from the server's list, not from the task's name", () => {
    expect(tourStatus("regression", null)).toBe("loading");
    expect(tourStatus("regression", ["classification", "regression"])).toBe("supported");
    expect(tourStatus("regression", ["classification"])).toBe("missing");
    expect(tourStatus("my_custom_task", ["classification", "regression"])).toBe("missing");
  });

  it("always asks the server for a researcher's own task, whose tour is not in the list", () => {
    expect(tourSupport("my_custom_task", null, true)).toBe("supported");
    expect(tourSupport("my_custom_task", ["classification"], true)).toBe("supported");
    expect(tourSupport("my_custom_task", ["classification"], false)).toBe("missing");
    expect(tourSupport("regression", null, false)).toBe("loading");
  });

  it("shows the server's reason when it has no tour for this run", () => {
    expect(unavailableReason(new ApiError(404, "Só o backend torchvision."))).toBe("Só o backend torchvision.");
    expect(unavailableReason(new ApiError(500, "boom"))).toBeNull();
    expect(unavailableReason(new Error("offline"))).toBeNull();
  });

  it("words line ranges, and the missing tour in both languages", () => {
    expect(pt.codeTour.lines(10, 10)).toBe("linha 10");
    expect(pt.codeTour.lines(10, 14)).toBe("linhas 10–14");
    expect(pt.codeTour.missing).not.toBe(en.codeTour.missing);
  });

  it("says its grain is the step, not the line", () => {
    expect(tab("pt")).toContain(pt.codeTour.granularity);
    expect(tab("en")).toContain(en.codeTour.granularity);
  });
});

describe("the drawing", () => {
  it("draws the family of the run, small, with the caption that its motion is illustrative", () => {
    const familyOf = (task: string, config: unknown) =>
      tab("pt", { spec: storySpecFor(task, config) }).match(/data-family="([a-z_]+)"/)?.[1];
    expect(familyOf("classification", { model: { name: "swin_t" } })).toBe("transformer");
    expect(familyOf("segmentation", { model: { name: "unet" } })).toBe("encoder_decoder");
    expect(familyOf("anomaly", { model: { name: "patchcore" } })).toBe("patchcore");
    expect(tab("pt")).toContain(pt.trainingStory.caption);
    expect(pt.trainingStory.caption).toMatch(/ilustrativa, em câmera lenta/);
  });

  it("keeps the animation off a sheet that is minimized", () => {
    expect(tab("pt")).toContain("animateMotion");
    expect(tab("pt", { sheetVisible: false })).not.toContain("animateMotion");
  });
});
