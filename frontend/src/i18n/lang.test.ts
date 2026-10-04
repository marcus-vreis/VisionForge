import { describe, expect, it } from "vitest";
import { initialLang } from "./lang";

describe("initialLang", () => {
  it("keeps what the user chose before", () => {
    expect(initialLang("en", "pt-BR")).toBe("en");
    expect(initialLang("pt", "en-US")).toBe("pt");
  });

  it("follows the browser on the first visit", () => {
    expect(initialLang(null, "pt-BR")).toBe("pt");
    expect(initialLang(null, "pt-PT")).toBe("pt");
    expect(initialLang(null, "en-GB")).toBe("en");
    expect(initialLang(null, "de-DE")).toBe("en");
  });

  it("ignores a stored value it does not know", () => {
    expect(initialLang("fr", "pt-BR")).toBe("pt");
  });

  it("falls back to English with no browser hint", () => {
    expect(initialLang(null, undefined)).toBe("en");
  });
});
