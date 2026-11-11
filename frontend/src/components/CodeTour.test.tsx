import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ApiError } from "../api/client";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { libraryNotice, tourStatus, unavailableReason } from "../lib/code-tour";
import { SHEET_TABS } from "../lib/training-story";
import { CodeTour } from "./CodeTour";

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };
const wrap = (element: React.ReactNode) =>
  renderToStaticMarkup(<LanguageContext.Provider value={lang}>{element}</LanguageContext.Provider>);

describe("CodeTour", () => {
  it("is the last tab of the training sheet", () => {
    expect(SHEET_TABS).toEqual(["log", "results", "story", "code"]);
  });
  it("waits for the server's list of tasks before saying a task has no tour", () => {
    const html = wrap(<CodeTour taskKey="regression" settings={{ loss: "mae" }} accent="red" />);
    expect(html).toContain(pt.codeTour.loading);
    expect(html).not.toContain(pt.codeTour.missing);
  });
  it("decides support from the server's list, not from the task's name", () => {
    expect(tourStatus("regression", null)).toBe("loading");
    expect(tourStatus("regression", ["classification", "regression"])).toBe("supported");
    expect(tourStatus("regression", ["classification"])).toBe("missing");
    expect(tourStatus("my_custom_task", ["classification", "regression"])).toBe("missing");
  });
  it("shows the server's reason when it has no tour for this run", () => {
    expect(unavailableReason(new ApiError(404, "Só o backend torchvision."))).toBe("Só o backend torchvision.");
    expect(unavailableReason(new ApiError(500, "boom"))).toBeNull();
    expect(unavailableReason(new Error("offline"))).toBeNull();
  });
  it("words line ranges", () => {
    expect(pt.codeTour.lines(10, 10)).toBe("linha 10");
    expect(pt.codeTour.lines(10, 14)).toBe("linhas 10–14");
  });
  it("marks a library's code with its version, license and resolved notes", () => {
    const section = {
      id: "yolo_loop", name: "BaseTrainer._do_train", file: "ultralytics/engine/trainer.py",
      first_line: 376, lines: ["def _do_train(self):"], notes: [],
      notes_resolved: 9, notes_total: 10,
      library: { name: "ultralytics", version: "8.4.60", license: "AGPL-3.0" },
    };
    const notice = libraryNotice(section, pt.codeTour);
    expect(notice).toContain("ultralytics 8.4.60");
    expect(notice).toContain("AGPL-3.0");
    expect(notice).toContain("9 de 10");
    expect(libraryNotice({ ...section, library: null }, pt.codeTour)).toBeNull();
    expect(libraryNotice(section, en.codeTour)).toContain("9 of 10");
  });
  it("titles the Ultralytics sections in both languages", () => {
    for (const id of ["yolo_args", "yolo_augment", "yolo_fit", "yolo_epoch", "yolo_loop"]) {
      expect(pt.codeTour.sections[id]).toBeTruthy();
      expect(en.codeTour.sections[id]).toBeTruthy();
    }
  });
  it("names the missing tour in both languages", () => {
    expect(pt.codeTour.missing).not.toBe(en.codeTour.missing);
  });
});
