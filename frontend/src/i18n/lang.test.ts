import { describe, expect, it } from "vitest";
import { initialLang, localeOf, readStoredLang, storeLang } from "./lang";

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

describe("localeOf", () => {
  it("gives each language its BCP 47 tag", () => {
    expect(localeOf("pt")).toBe("pt-BR");
    expect(localeOf("en")).toBe("en-US");
  });
});

describe("storage", () => {
  // These run in vitest's node environment, which has no localStorage: both
  // functions have to swallow that instead of taking the page down with them.
  it("reads nothing when storage is unavailable", () => {
    expect(readStoredLang()).toBeNull();
  });

  it("does not throw when storage is unavailable", () => {
    expect(() => storeLang("en")).not.toThrow();
  });
});
