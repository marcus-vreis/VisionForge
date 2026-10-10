import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import { en } from "../../i18n/en";
import { pt } from "../../i18n/pt";
import { GUIDES, guideById } from ".";
import { folderTail } from "./first-training";
import type { GuideContext } from "./types";

const SRC = join(__dirname, "..", "..");

/** Every `data-tour="…"` the components actually render. */
function markedAnchors(): Set<string> {
  const found = new Set<string>();
  const walk = (dir: string) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (/\.tsx$/.test(entry.name) && !entry.name.endsWith(".test.tsx")) {
        for (const m of readFileSync(path, "utf8").matchAll(/data-tour="([\w-]+)"/g)) {
          found.add(m[1]);
        }
      }
    }
  };
  walk(SRC);
  return found;
}

describe("the registry", () => {
  it("lists the tour first and Primeiro treino second", () => {
    expect(GUIDES.map((g) => g.id)).toEqual(["tour", "firstTraining"]);
  });

  it("finds each guide by id", () => {
    for (const g of GUIDES) expect(guideById(g.id)).toBe(g);
  });

  it("gives every guide a title, a summary and steps in both languages", () => {
    for (const dict of [pt, en]) {
      for (const g of GUIDES) {
        expect(g.title(dict).length).toBeGreaterThan(3);
        expect(g.summary(dict).length).toBeGreaterThan(10);
        expect(g.steps(dict).length).toBeGreaterThanOrEqual(7);
      }
    }
  });

  it("names a title that differs between the two guides", () => {
    const titles = GUIDES.map((g) => g.title(pt));

    expect(new Set(titles).size).toBe(titles.length);
  });

  it("points only at anchors the components mark", () => {
    const marked = markedAnchors();

    for (const g of GUIDES) {
      for (const step of g.steps(pt)) {
        if (step.anchor) expect(marked, `${g.id}: ${step.anchor}`).toContain(step.anchor);
      }
    }
  });
});

describe("Primeiro treino", () => {
  const steps = (dict: typeof pt) => guideById("firstTraining").steps(dict);

  it("has eight steps, each short enough to read at a glance", () => {
    for (const dict of [pt, en]) {
      expect(steps(dict)).toHaveLength(8);
      for (const step of steps(dict)) {
        expect(step.title.length).toBeGreaterThan(0);
        expect(step.body.length).toBeGreaterThan(40);
        // 1-3 sentences, not an essay (the last step is a list, so it gets more room).
        expect(step.body.length).toBeLessThan(560);
      }
    }
  });

  it("opens on the classification tab and puts the sheet away before the History", () => {
    const calls: string[] = [];
    const ctx: GuideContext = {
      selectTask: (k) => calls.push(`task:${k}`),
      setDatasetPath: (p) => calls.push(`path:${p}`),
      hideTrainingSheet: () => calls.push("hide"),
    };
    const all = steps(pt);

    all[0].onEnter?.(ctx);
    all[6].onEnter?.(ctx);

    expect(calls).toEqual(["task:classification", "hide"]);
  });

  it("states that the synthetic data is easy on purpose, in both languages", () => {
    const result = (dict: typeof pt) => steps(dict)[5].body.toLowerCase();

    expect(result(pt)).toMatch(/de propósito/);
    expect(result(en)).toMatch(/on purpose/);
  });

  it("covers every other task in its last step, and a real dataset", () => {
    expect(steps(pt)[7].body).toMatch(/YOLO[\s\S]*mAP[\s\S]*mIoU[\s\S]*R²[\s\S]*RMSE[\s\S]*AUROC/);
    expect(steps(en)[7].body).toMatch(/YOLO[\s\S]*mAP[\s\S]*mIoU[\s\S]*R²[\s\S]*RMSE[\s\S]*AUROC/);
    expect(steps(pt)[7].anchor).toBe("datasets");
  });
});

describe("the sample dataset action", () => {
  afterEach(() => vi.unstubAllGlobals());

  const ctx = () => {
    const paths: string[] = [];
    const context: GuideContext = {
      selectTask: () => {},
      setDatasetPath: (p) => paths.push(p),
      hideTrainingSheet: () => {},
    };
    return { context, paths };
  };
  const action = () => guideById("firstTraining").steps(pt)[0].action!;
  const reply = (status: number, body: unknown) =>
    vi.fn().mockResolvedValue(
      new Response(JSON.stringify(body), {
        status,
        headers: { "Content-Type": "application/json" },
      }),
    );

  it("creates the dataset and fills the field with its path", async () => {
    const fetchMock = reply(201, {
      path: "C:/work/datasets/exemplo-classificacao",
      classes: ["class_a", "class_b"],
      counts: { train: 12, val: 12, test: 12 },
    });
    vi.stubGlobal("fetch", fetchMock);
    const { context, paths } = ctx();

    const out = await action().run(context);

    expect(out.ok).toBe(true);
    expect(paths).toEqual(["C:/work/datasets/exemplo-classificacao"]);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/sample-dataset");
    expect(JSON.parse((init as RequestInit).body as string)).toEqual({
      task: "classification",
    });
  });

  it("reuses the folder that already exists when the server answers 409", async () => {
    vi.stubGlobal(
      "fetch",
      reply(409, { detail: "already exists", path: "C:/work/datasets/exemplo-classificacao" }),
    );
    const { context, paths } = ctx();

    const out = await action().run(context);

    expect(out.ok).toBe(true);
    expect(out.message).toContain("datasets/exemplo-classificacao");
    expect(paths).toEqual(["C:/work/datasets/exemplo-classificacao"]);
  });

  it("names the folder by its last two segments, on either kind of slash", () => {
    expect(folderTail("C:\\work\\datasets\\exemplo-classificacao")).toBe(
      "datasets/exemplo-classificacao",
    );
    expect(folderTail("/home/ana/datasets/exemplo-classificacao/")).toBe(
      "datasets/exemplo-classificacao",
    );
    expect(folderTail("solo")).toBe("solo");
  });

  it("shows the failure on the card and leaves the field alone", async () => {
    vi.stubGlobal("fetch", reply(500, { detail: "Could not write the sample dataset: disk full" }));
    const { context, paths } = ctx();

    const out = await action().run(context);

    expect(out.ok).toBe(false);
    expect(out.message).toContain("disk full");
    expect(paths).toEqual([]);
  });

  it("does not mistake a 409 without a path for a reusable folder", async () => {
    vi.stubGlobal("fetch", reply(409, { detail: "busy" }));
    const { context, paths } = ctx();

    const out = await action().run(context);

    expect(out.ok).toBe(false);
    expect(paths).toEqual([]);
  });
});
