import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ApiError } from "../api/client";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { tourStatus, unavailableReason } from "../lib/code-tour";
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
  it("names the missing tour in both languages", () => {
    expect(pt.codeTour.missing).not.toBe(en.codeTour.missing);
  });
});
