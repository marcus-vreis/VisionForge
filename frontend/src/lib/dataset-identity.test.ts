import { describe, expect, it } from "vitest";

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
    expect(compareDatasets(info(), info()).kind).toBe("same");
  });

  it("reports different data when the digests differ", () => {
    expect(compareDatasets(info(), info({ digest: "999" })).kind).toBe("different");
  });

  it("refuses to answer when one run has no digest", () => {
    // 26 of the 28 runs on disk are in this state, so this is the common path.
    const verdict = compareDatasets(info(), info({ digest: null }));

    expect(verdict.kind).toBe("unknown");
    expect(verdict.kind === "unknown" && verdict.reason).toMatch(/fingerprint/i);
  });

  it("refuses to answer when the two used different methods", () => {
    // A manifest digest and a content digest of the same data do not match;
    // calling that "different data" would be a lie.
    const verdict = compareDatasets(info(), info({ method: "content" }));

    expect(verdict.kind).toBe("unknown");
    expect(verdict.kind === "unknown" && verdict.reason).toMatch(/método/i);
  });

  it("refuses to answer when a run has no dataset at all", () => {
    expect(compareDatasets(info(), null).kind).toBe("unknown");
  });
});

describe("formatBytes", () => {
  it("scales to a readable unit", () => {
    expect(formatBytes(123456789)).toBe("117,7 MB");
  });

  it("leaves bytes unscaled", () => {
    expect(formatBytes(512)).toBe("512 B");
  });

  it("handles a missing size", () => {
    expect(formatBytes(null)).toBe("—");
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
