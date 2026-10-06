import { describe, expect, it } from "vitest";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { pickerCancelText, SERVER_CANCEL_MESSAGE } from "./picker-feedback";

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
