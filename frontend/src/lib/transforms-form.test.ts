import { describe, expect, it } from "vitest";

import {
  buildTransformsPayload,
  makeDefaultTransformsForm,
  parseTriple,
} from "./transforms-form";

describe("augment flag", () => {
  it("defaults to on, matching the backend", () => {
    expect(makeDefaultTransformsForm().augment).toBe(true);
  });

  it("travels in the payload so run.json records it", () => {
    const form = { ...makeDefaultTransformsForm(), augment: false };

    expect(buildTransformsPayload(form).augment).toBe(false);
  });

  it("keeps the tuned values when off, so turning it back on restores them", () => {
    // The whole reason this is a flag and not "write neutral values".
    const form = { ...makeDefaultTransformsForm(), augment: false, rotation_degrees: 25 };

    const payload = buildTransformsPayload(form);

    expect(payload.rotation_degrees).toBe(25);
    expect(payload.horizontal_flip).toBe(true);
  });
});

describe("parseTriple", () => {
  it("parses a well-formed triple", () => {
    expect(parseTriple("0.1, 0.2, 0.3", [1, 1, 1])).toEqual([0.1, 0.2, 0.3]);
  });

  it("falls back rather than producing a 422 on a typo", () => {
    expect(parseTriple("0.1, oops", [1, 1, 1])).toEqual([1, 1, 1]);
  });
});
