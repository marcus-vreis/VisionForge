import { describe, expect, it } from "vitest";

import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { compareDatasets, formatBytes, shortDigest } from "./dataset-identity";
import type { DatasetInfo } from "../types/run";

function info(overrides: Partial<DatasetInfo> = {}): DatasetInfo {
  return {
    name: "USK-COFFEE",
    root: "C:/data/USK-COFFEE",
    n_files: 8000,
    total_bytes: 123456789,
    method: "manifest",
    digest: "abc123def456789",
    note: "paths+sizes only",
    ...overrides,
  };
}

describe("compareDatasets", () => {
  it("reports the same data when both digests match", () => {
    expect(compareDatasets(pt, info(), info()).kind).toBe("same");
  });

  it("reports different data when the digests differ", () => {
    expect(compareDatasets(pt, info(), info({ digest: "999" })).kind).toBe("different");
  });

  it("refuses to answer when one run has no digest", () => {
    // 26 of the 28 runs on disk are in this state, so this is the common path.
    const verdict = compareDatasets(pt, info(), info({ digest: null }));

    expect(verdict.kind).toBe("unknown");
    expect(verdict.kind === "unknown" && verdict.reason).toMatch(/fingerprint/i);
    const english = compareDatasets(en, info(), info({ digest: null }));
    expect(english.kind === "unknown" && english.reason).toMatch(/no fingerprint/i);
  });

  it("refuses to answer when the two used different methods", () => {
    // A manifest digest and a content digest of the same data do not match;
    // calling that "different data" would be a lie.
    const verdict = compareDatasets(pt, info(), info({ method: "content" }));

    expect(verdict.kind).toBe("unknown");
    expect(verdict.kind === "unknown" && verdict.reason).toMatch(/método/i);
    const english = compareDatasets(en, info(), info({ method: "content" }));
    expect(english.kind === "unknown" && english.reason).toMatch(/different method/i);
  });

  it("refuses to answer when a run has no dataset at all", () => {
    expect(compareDatasets(pt, info(), null).kind).toBe("unknown");
  });
});

describe("formatBytes", () => {
  it("scales to a readable unit", () => {
    expect(formatBytes(123456789, "pt-BR")).toBe("117,7 MB");
  });

  it("puts the decimal separator where the locale does", () => {
    expect(formatBytes(123456789, "en-US")).toBe("117.7 MB");
    expect(formatBytes(910_848, "pt-BR")).toBe("889,5 KB");
    expect(formatBytes(910_848, "en-US")).toBe("889.5 KB");
  });

  it("leaves bytes unscaled", () => {
    expect(formatBytes(512, "pt-BR")).toBe("512 B");
    expect(formatBytes(512, "en-US")).toBe("512 B");
  });

  it("never groups digits, however large the size", () => {
    expect(formatBytes(1500 * 1024 ** 4, "en-US")).toBe("1500.0 TB");
  });

  it("handles a missing size", () => {
    expect(formatBytes(null, "pt-BR")).toBe("—");
  });
});

describe("shortDigest", () => {
  it("keeps the first 12 characters", () => {
    expect(shortDigest("abc123def456789")).toBe("abc123def456");
  });

  it("handles a missing digest", () => {
    expect(shortDigest(null)).toBe("—");
  });
});
