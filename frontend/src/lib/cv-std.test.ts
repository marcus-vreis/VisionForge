import { describe, expect, it } from "vitest";

import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { stdDdof } from "./cv-std";

describe("stdDdof", () => {
  it("reads the divisor convention a K-fold run recorded", () => {
    expect(stdDdof({ std_ddof: 1 })).toBe(1);
    expect(stdDdof({ std_ddof: 0 })).toBe(0);
  });

  it("reads a run from before the field as the population std it used", () => {
    // Old run.json files divide by n; the field only exists once they divide by n-1.
    expect(stdDdof({})).toBe(0);
    expect(stdDdof(undefined)).toBe(0);
    expect(stdDdof(null)).toBe(0);
  });

  it("does not take anything else for the sample std", () => {
    expect(stdDdof({ std_ddof: "1" })).toBe(0);
    expect(stdDdof({ std_ddof: 2 })).toBe(0);
  });
});

describe("the K-fold std footnote", () => {
  it("names the convention, in both languages", () => {
    expect(pt.runDetail.cv.stdNote(1)).toMatch(/n−1|n-1/);
    expect(pt.runDetail.cv.stdNote(0)).toMatch(/divide por n|população|populacional/i);
    expect(en.runDetail.cv.stdNote(1)).toMatch(/n−1|n-1/);
    expect(en.runDetail.cv.stdNote(0)).toMatch(/divides by n|population/i);
  });
});
