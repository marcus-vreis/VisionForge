import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { SAMPLE_SETUP } from "../lib/guides/first-training";
import { GUIDE_TASKS } from "../lib/guides/tasks";
import { runSample } from "../lib/sample-dataset";
import { SampleDatasetButton } from "./SampleDatasetButton";

const reply = (status: number, json: unknown) =>
  vi.fn().mockResolvedValue(
    new Response(JSON.stringify(json), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );

const failed = (r: string) => `falhou: ${r}`;

describe("runSample", () => {
  afterEach(() => vi.unstubAllGlobals());

  it.each(GUIDE_TASKS)("asks for %s's sample and hands the path and setup to the form", async (task) => {
    const fetchMock = reply(201, {
      task,
      path: `C:/work/datasets/exemplo-${task}`,
      classes: [],
      targets: [],
      counts: {},
    });
    vi.stubGlobal("fetch", fetchMock);
    const calls: unknown[][] = [];

    const note = await runSample(task, (...a) => calls.push(a), failed);

    expect(note).toBeNull();
    expect(JSON.parse((fetchMock.mock.calls[0][1] as RequestInit).body as string)).toEqual({ task });
    expect(calls).toEqual([[`C:/work/datasets/exemplo-${task}`, SAMPLE_SETUP[task]]]);
  });

  it("reuses the path of a folder that already exists (409)", async () => {
    vi.stubGlobal("fetch", reply(409, { detail: "já existe", path: "C:/work/datasets/exemplo-segmentacao" }));
    const calls: unknown[][] = [];

    const note = await runSample("segmentation", (...a) => calls.push(a), failed);

    expect(note).toBeNull();
    expect(calls).toEqual([
      ["C:/work/datasets/exemplo-segmentacao", { numClasses: 3, imageSize: 64 }],
    ]);
  });

  it("returns the error sentence and leaves the form alone when the call fails", async () => {
    vi.stubGlobal("fetch", reply(500, { detail: "disco cheio" }));
    const calls: unknown[][] = [];

    const note = await runSample("anomaly", (...a) => calls.push(a), failed);

    expect(note).toMatch(/^falhou: /);
    expect(calls).toEqual([]);
  });
});

describe("SampleDatasetButton", () => {
  it.each(["pt", "en"] as const)("shows the button in %s, with no error before a click", (lang) => {
    const dict = lang === "pt" ? pt : en;
    const html = renderToStaticMarkup(
      <LanguageContext.Provider
        value={{ lang, t: dict, locale: lang === "pt" ? "pt-BR" : "en-US", setLang: () => {} }}
      >
        <SampleDatasetButton task="segmentation" onApply={() => {}} />
      </LanguageContext.Provider>,
    );

    expect(html).toContain(dict.sampleDataset.button);
    expect(html).toContain('data-sample-task="segmentation"');
    expect(html).not.toContain('role="alert"');
  });
});
