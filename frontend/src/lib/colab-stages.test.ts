import { describe, expect, it } from "vitest";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import type { TrainingEvent } from "../types/run";
import { elapsedTag, remoteLogLine } from "./colab-stages";
import { logLinesFor } from "./training-log";

const remote = (stage: string, extra: Record<string, unknown> = {}): TrainingEvent => ({
  event: "remote",
  stage,
  ...extra,
});

const text = (event: TrainingEvent, dict = pt): string => {
  const line = remoteLogLine(event);
  if (line === null) throw new Error("no line");
  return typeof line === "string" ? line : line(dict);
};

describe("elapsedTag", () => {
  it("writes minutes and seconds, and hours past the first", () => {
    expect(elapsedTag(0)).toBe("[0:00]");
    expect(elapsedTag(65.4)).toBe("[1:05]");
    expect(elapsedTag(1799.9)).toBe("[29:59]");
    expect(elapsedTag(3725)).toBe("[1:02:05]");
  });

  it("is empty for a server that sends no time", () => {
    expect(elapsedTag(undefined)).toBe("");
    expect(elapsedTag("12")).toBe("");
  });
});

describe("remoteLogLine", () => {
  it("says each stage of the VM with the time since the click", () => {
    expect(text(remote("allocated", { accelerator: "T4", elapsed_s: 42 }))).toBe(
      "> [0:42] VM do Colab alocada (T4)",
    );
    expect(text(remote("install", { version: "0.22.3", elapsed_s: 70 }))).toBe(
      "> [1:10] instalando o VisionForge 0.22.3 na VM",
    );
    expect(text(remote("drive_copy", { files: 600, seconds: 31.5, elapsed_s: 200 }))).toBe(
      "> [3:20] dataset copiado do Drive: 600 arquivos em 31.5 s",
    );
    expect(text(remote("released", { elapsed_s: 1800 }))).toBe("> [30:00] VM do Colab liberada");
  });

  it("follows the language", () => {
    expect(text(remote("train", { device: "cuda", elapsed_s: 5 }), en)).toBe("> [0:05] training (cuda)");
  });

  it("names a failed export to the Drive with its reason", () => {
    expect(text(remote("drive_export", { error: "quota" }))).toBe(
      "> não foi possível salvar o run no Drive: quota",
    );
  });

  it("shows an unknown stage as received instead of dropping it", () => {
    expect(text(remote("warming_up", { elapsed_s: 3 }))).toBe("> [0:03] colab: warming_up");
  });

  it("is null for anything that is not a VM stage", () => {
    expect(remoteLogLine({ event: "end", total_epochs: 1 })).toBeNull();
  });
});

describe("logLinesFor with a Colab job", () => {
  it("puts a line per stage between the epochs", () => {
    const { lines } = logLinesFor(
      [remote("kernel_ready", { elapsed_s: 90 }), remote("dataset", { elapsed_s: 400 })],
      null,
    );
    expect(lines.map((l) => (typeof l === "string" ? l : l(pt)))).toEqual([
      "> [1:30] VM pronta para receber o job",
      "> [6:40] preparando o dataset na VM",
    ]);
  });
});
