import { describe, expect, it, vi } from "vitest";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { pickerCancelText, runPicker, SERVER_CANCEL_MESSAGE } from "./picker-feedback";

describe("pickerCancelText", () => {
  it("words a plain dismissal in the language of the interface", () => {
    const res = { message: SERVER_CANCEL_MESSAGE };
    expect(pickerCancelText(res, en.datasetPicker.cancelled)).toBe("Selection canceled.");
    expect(pickerCancelText(res, pt.datasetPicker.cancelled)).toBe(pt.datasetPicker.cancelled);
    expect(pickerCancelText(res, en.paramPanel.weights.cancelled)).toBe("Canceled.");
    expect(pickerCancelText(res, en.runDetail.picker.cancelled)).toBe("Canceled.");
  });

  it("falls back to the dictionary when the server sent no message", () => {
    expect(pickerCancelText({ message: null }, en.runDetail.picker.cancelled)).toBe("Canceled.");
  });

  it("keeps a message that says why the dialog did not open", () => {
    expect(
      pickerCancelText({ message: "Falha ao abrir o seletor: boom" }, en.runDetail.picker.cancelled),
    ).toBe("Falha ao abrir o seletor: boom");
    expect(
      pickerCancelText(
        { message: "tkinter is not available on this Python installation." },
        en.runDetail.picker.cancelled,
      ),
    ).toBe("tkinter is not available on this Python installation.");
  });

  it("matches the sentence the backend sends", () => {
    expect(SERVER_CANCEL_MESSAGE).toBe("Cancelado.");
  });
});

describe("runPicker", () => {
  const texts = { canceled: "Selection canceled.", failed: "Failed to open the folder picker." };

  it("hands a chosen path over and shows no notice", async () => {
    const onPath = vi.fn();
    const note = await runPicker(
      async () => ({ cancelled: false, path: "/data/coffee", message: null }),
      onPath,
      texts,
    );
    expect(onPath).toHaveBeenCalledWith("/data/coffee");
    expect(note).toBeNull();
  });

  it("says so when the dialog is dismissed", async () => {
    const onPath = vi.fn();
    const note = await runPicker(
      async () => ({ cancelled: true, path: "", message: SERVER_CANCEL_MESSAGE }),
      onPath,
      texts,
    );
    expect(onPath).not.toHaveBeenCalled();
    expect(note).toBe("Selection canceled.");
  });

  it("shows why the dialog could not open", async () => {
    const note = await runPicker(
      async () => ({ cancelled: true, path: "", message: "Falha ao abrir o seletor: boom" }),
      vi.fn(),
      texts,
    );
    expect(note).toBe("Falha ao abrir o seletor: boom");
  });

  it("shows the error of a failed request, or the fallback when it has none", async () => {
    const boom = await runPicker(async () => Promise.reject(new Error("HTTP 500")), vi.fn(), texts);
    expect(boom).toBe("HTTP 500");
    const bare = await runPicker(async () => Promise.reject("x"), vi.fn(), texts);
    expect(bare).toBe(texts.failed);
  });
});
