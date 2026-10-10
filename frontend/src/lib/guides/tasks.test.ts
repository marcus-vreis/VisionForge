import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import { en } from "../../i18n/en";
import { pt } from "../../i18n/pt";
import { GUIDE_TASKS, guideById, guideTaskFor, type GuideTask } from ".";
import { SAMPLE_SETUP } from "./first-training";
import { gateOpen } from "./gates";
import type { GuideContext, GuideFacts, GuideStep } from "./types";

const COMPONENTS = join(__dirname, "..", "..", "components");

/** The files whose panel carries each anchor of the script, per task. */
const PANELS: Record<GuideTask, { dataset: string; training: string }> = {
  classification: { dataset: "DatasetPicker.tsx", training: "ParamPanel.tsx" },
  detection: { dataset: "DetectionPanel.tsx", training: "DetectionPanel.tsx" },
  regression: { dataset: "RegressionPanel.tsx", training: "RegressionPanel.tsx" },
  segmentation: { dataset: "SegmentationPanel.tsx", training: "SegmentationPanel.tsx" },
  anomaly: { dataset: "AnomalyPanel.tsx", training: "AnomalyPanel.tsx" },
};

const steps = (dict: typeof pt, task: GuideTask): GuideStep[] =>
  guideById("firstTraining").steps(dict, task);

const facts = (over: Partial<GuideFacts> = {}): GuideFacts => ({
  datasetPath: "",
  trainingStarted: false,
  trainingEnded: false,
  ...over,
});

describe("which task a guide plays", () => {
  it("is the active tab when it is a built-in task", () => {
    for (const task of GUIDE_TASKS) expect(guideTaskFor(task)).toBe(task);
  });

  it("falls back to classification for a researcher's own task", () => {
    expect(guideTaskFor("example_counting")).toBe("classification");
    expect(guideTaskFor("")).toBe("classification");
  });

  it("knows the five built-in tasks", () => {
    expect([...GUIDE_TASKS].sort()).toEqual(
      ["anomaly", "classification", "detection", "regression", "segmentation"],
    );
  });
});

describe.each(GUIDE_TASKS)("Primeiro treino: %s", (task) => {
  it("has the same eight stops in both languages, each with text", () => {
    for (const dict of [pt, en]) {
      const all = steps(dict, task);
      expect(all).toHaveLength(8);
      for (const step of all) {
        expect(step.title.trim().length).toBeGreaterThan(0);
        expect(step.body.length).toBeGreaterThan(40);
        expect(step.body.length).toBeLessThan(560);
      }
    }
  });

  it("points at anchors the task's own panel marks", () => {
    const marks = (file: string, anchor: string) =>
      readFileSync(join(COMPONENTS, file), "utf8").includes(`data-tour="${anchor}"`);
    const [sample, found, parameters, train] = steps(pt, task);

    expect(sample.anchor).toBe("dataset");
    expect(marks(PANELS[task].dataset, "dataset")).toBe(true);
    expect(found.anchor).toBe(task === "classification" ? "dataset" : "dataset-stats");
    expect(marks(PANELS[task].dataset, found.anchor!)).toBe(true);
    expect(parameters.anchor).toBe("training");
    expect(marks(PANELS[task].training, "training")).toBe(true);
    expect(train.anchor).toBe("train");
    expect(marks("BottomBar.tsx", "train")).toBe(true);
    expect(marks("TrainingOverlay.tsx", "training-sheet")).toBe(true);
  });

  it("uses the same anchors in both languages, and the same ones as classification bar the stats", () => {
    const anchors = (t: GuideTask) => steps(pt, t).map((s) => s.anchor);
    const bar = (list: (string | undefined)[]) => list.filter((_, i) => i !== 1);

    expect(steps(en, task).map((s) => s.anchor)).toEqual(anchors(task));
    expect(bar(anchors(task))).toEqual(bar(anchors("classification")));
  });

  it("gates the dataset, the start and the end, and nothing else", () => {
    const all = steps(pt, task);
    const gated = all.flatMap((s, i) => (s.waitFor ? [i] : []));

    expect(gated).toEqual([1, 3, 5]);
    expect(gateOpen(all[1], facts())).toBe(false);
    expect(gateOpen(all[1], facts({ datasetPath: "  " }))).toBe(false);
    expect(gateOpen(all[1], facts({ datasetPath: "C:/work/datasets/x" }))).toBe(true);
    expect(gateOpen(all[3], facts({ datasetPath: "x" }))).toBe(false);
    expect(gateOpen(all[3], facts({ trainingStarted: true }))).toBe(true);
    expect(gateOpen(all[5], facts({ trainingStarted: true }))).toBe(false);
    expect(gateOpen(all[5], facts({ trainingStarted: true, trainingEnded: true }))).toBe(true);
    for (const i of gated) {
      expect(all[i].waitHint?.length).toBeGreaterThan(10);
      expect(steps(en, task)[i].waitHint?.length).toBeGreaterThan(10);
    }
  });

  it("opens on its own tab and puts the sheet away before the History", () => {
    const calls: string[] = [];
    const ctx: GuideContext = {
      selectTask: (k) => calls.push(`task:${k}`),
      setDatasetPath: () => {},
      hideTrainingSheet: () => calls.push("hide"),
    };
    const all = steps(pt, task);

    all[0].onEnter?.(ctx);
    all[6].onEnter?.(ctx);

    expect(calls).toEqual([`task:${task}`, "hide"]);
  });

  it("says the synthetic data is easy on purpose, and points at the download card", () => {
    const text = (dict: typeof pt) => steps(dict, task).map((s) => s.body).join("\n");

    expect(text(pt)).toMatch(/de propósito/);
    expect(text(en)).toMatch(/on purpose/);
    expect(steps(pt, task)[7].body).toMatch(/cartão de download/);
    expect(steps(en, task)[7].body).toMatch(/download card/);
    expect(steps(pt, task)[7].anchor).toBe("datasets");
  });

  it("gives the task its own text for what the data is, the parameters and the result", () => {
    for (const dict of [pt, en]) {
      for (const i of [0, 1, 2, 4, 5, 7]) {
        for (const other of GUIDE_TASKS.filter((t) => t !== task)) {
          expect(steps(dict, task)[i].body, `step ${i} vs ${other}`).not.toBe(
            steps(dict, other)[i].body,
          );
        }
      }
    }
  });
});

describe("what each task's script teaches", () => {
  const body = (dict: typeof pt, task: GuideTask, i: number) => steps(dict, task)[i].body;

  it("names the data of each task", () => {
    expect(body(pt, "detection", 0)).toMatch(/YOLO/);
    expect(body(en, "detection", 0)).toMatch(/YOLO/);
    expect(body(pt, "regression", 0)).toMatch(/CSV/);
    expect(body(en, "regression", 0)).toMatch(/CSV/);
    expect(body(pt, "segmentation", 0)).toMatch(/máscara/);
    expect(body(en, "segmentation", 0)).toMatch(/mask/);
    expect(body(pt, "anomaly", 0)).toMatch(/só com imagens normais/);
    expect(body(en, "anomaly", 0)).toMatch(/normal images only/);
  });

  it("names the metric of each task in the result step", () => {
    for (const dict of [pt, en]) {
      expect(body(dict, "detection", 5)).toMatch(/mAP50-95/);
      expect(body(dict, "regression", 5)).toMatch(/R²[\s\S]*RMSE/);
      expect(body(dict, "segmentation", 5)).toMatch(/mIoU[\s\S]*Dice/);
      expect(body(dict, "anomaly", 5)).toMatch(/AUROC[\s\S]*F1/);
    }
  });

  it("tells anomaly detection that PatchCore has phases, not epochs", () => {
    expect(body(pt, "anomaly", 4)).toMatch(/PatchCore não há épocas[\s\S]*fases/);
    expect(body(en, "anomaly", 4)).toMatch(/PatchCore has no epochs[\s\S]*phases/);
  });

  it("warns that the test split of regression is small", () => {
    expect(body(pt, "regression", 5)).toMatch(/4 imagens/);
    expect(body(en, "regression", 5)).toMatch(/4 images/);
  });

  it("asks for the few values that make each sample train, and says what it sets itself", () => {
    expect(body(pt, "regression", 2)).toMatch(/30[\s\S]*batch[\s\S]*4/);
    expect(body(pt, "segmentation", 2)).toMatch(/épocas para 8/);
    expect(body(pt, "anomaly", 2)).toMatch(/épocas para 3/);
    expect(body(pt, "detection", 2)).toMatch(/deixe tudo como está/);
    expect(body(pt, "segmentation", 1)).toMatch(/3/);
    expect(body(en, "segmentation", 1)).toMatch(/3/);
    expect(SAMPLE_SETUP.segmentation).toEqual({ numClasses: 3, imageSize: 64 });
    expect(SAMPLE_SETUP.detection).toEqual({ numClasses: 1, imageSize: 128 });
    expect(SAMPLE_SETUP.regression).toEqual({ imageSize: 64 });
    expect(SAMPLE_SETUP.anomaly).toEqual({ imageSize: 64 });
    expect(SAMPLE_SETUP.classification).toEqual({});
  });
});

describe("the sample dataset action, per task", () => {
  afterEach(() => vi.unstubAllGlobals());

  const reply = (status: number, json: unknown) =>
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify(json), {
        status,
        headers: { "Content-Type": "application/json" },
      }),
    );

  it.each(GUIDE_TASKS)("asks the server for %s's sample and fills that task's form", async (task) => {
    const fetchMock = reply(201, {
      task,
      path: `C:/work/datasets/exemplo-${task}`,
      classes: [],
      targets: [],
      counts: {},
    });
    vi.stubGlobal("fetch", fetchMock);
    const calls: unknown[][] = [];
    const ctx: GuideContext = {
      selectTask: () => {},
      setDatasetPath: (...args) => calls.push(args),
      hideTrainingSheet: () => {},
    };

    const out = await steps(pt, task)[0].action!.run(ctx);

    expect(out.ok).toBe(true);
    expect(JSON.parse((fetchMock.mock.calls[0][1] as RequestInit).body as string)).toEqual({ task });
    expect(calls).toEqual([[`C:/work/datasets/exemplo-${task}`, task, SAMPLE_SETUP[task]]]);
  });

  it("applies the setup to a folder that already existed too", async () => {
    vi.stubGlobal("fetch", reply(409, { detail: "já existe", path: "C:/work/datasets/exemplo-segmentacao" }));
    const calls: unknown[][] = [];
    const ctx: GuideContext = {
      selectTask: () => {},
      setDatasetPath: (...args) => calls.push(args),
      hideTrainingSheet: () => {},
    };

    const out = await steps(pt, "segmentation")[0].action!.run(ctx);

    expect(out.ok).toBe(true);
    expect(out.message).toContain("datasets/exemplo-segmentacao");
    expect(calls).toEqual([
      ["C:/work/datasets/exemplo-segmentacao", "segmentation", { numClasses: 3, imageSize: 64 }],
    ]);
  });
});
