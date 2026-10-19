import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { THEME_KEY, parseThemePref, resolveTheme } from "./theme";
import { convert } from "../../scripts/theme-codemod.mjs";

describe("parseThemePref", () => {
  it("keeps a pinned theme and reads anything else as system", () => {
    expect(parseThemePref("light")).toBe("light");
    expect(parseThemePref("dark")).toBe("dark");
    expect(parseThemePref(null)).toBe("system");
    expect(parseThemePref("sepia")).toBe("system");
  });
});

describe("resolveTheme", () => {
  it("follows the operating system only when the choice is system", () => {
    expect(resolveTheme("system", true)).toBe("dark");
    expect(resolveTheme("system", false)).toBe("light");
    expect(resolveTheme("light", true)).toBe("light");
    expect(resolveTheme("dark", false)).toBe("dark");
  });
});

describe("the boot script in index.html", () => {
  it("reads the same storage key, so the first paint matches the switch", () => {
    const html = readFileSync(join(__dirname, "..", "..", "index.html"), "utf8");
    expect(html).toContain(`"${THEME_KEY}"`);
    expect(html).toContain("prefers-color-scheme: dark");
  });
});

describe("theme codemod", () => {
  it("keeps the dark value as the fallback of a light-only variable", () => {
    expect(convert('"rgba(255,255,255,0.04)"')).toBe('"rgb(var(--lt-ink, 255 255 255) / 0.04)"');
    expect(convert('"rgba(0,0,0,0.30)"')).toBe('"rgb(var(--lt-shade, 0 0 0) / 0.30)"');
    expect(convert('"rgba(12,14,18,0.65)"')).toBe('"rgb(var(--lt-surface, 12 14 18) / 0.65)"');
  });

  it("mirrors only solid tones light enough to have been picked for a dark background", () => {
    expect(convert("oklch(0.85 0.14 22)")).toBe("oklch(calc(0.85 - var(--lt-flip, 0) * 0.4) 0.14 22)");
    expect(convert("oklch(0.704 0.191 22.216)")).toBe("oklch(0.704 0.191 22.216)");
    expect(convert("oklch(0.85 0.14 22 / 0.10)")).toBe("oklch(0.85 0.14 22 / 0.10)");
  });

  it("leaves colored rgba and already converted values alone", () => {
    const done = "rgb(var(--lt-ink, 255 255 255) / 0.04) rgba(180, 140, 255, 0.12)";
    expect(convert(done)).toBe(done);
  });
});
