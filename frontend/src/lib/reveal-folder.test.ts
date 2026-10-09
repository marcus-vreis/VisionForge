import { describe, expect, it } from "vitest";

import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { canOfferReveal, revealErrorText } from "./reveal-folder";

describe("canOfferReveal", () => {
  it("offers the button only when the server said it can open the folder", () => {
    expect(canOfferReveal({ can_reveal: true })).toBe(true);
    expect(canOfferReveal({ can_reveal: false })).toBe(false);
  });

  it("does not offer it while the run is loading or when the server omits the flag", () => {
    expect(canOfferReveal(null)).toBe(false);
    expect(canOfferReveal(undefined)).toBe(false);
    expect(canOfferReveal({})).toBe(false);
  });
});

describe("revealErrorText", () => {
  const texts = { forbidden: "403", notFound: "404", failed: "other" };

  it("names the refusal and the missing run, and falls back for anything else", () => {
    expect(revealErrorText(403, texts)).toBe("403");
    expect(revealErrorText(404, texts)).toBe("404");
    expect(revealErrorText(500, texts)).toBe("other");
    expect(revealErrorText(0, texts)).toBe("other");
  });

  it("has a distinct sentence for each case in both languages", () => {
    for (const dict of [pt, en]) {
      const r = dict.runDetail.reveal;
      expect(new Set([r.forbidden, r.notFound, r.failed]).size).toBe(3);
    }
  });
});
