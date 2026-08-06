import { describe, expect, it } from "vitest";

import {
  currentBundleName,
  detectStaleServer,
  isServerStale,
  type HealthResponse,
} from "./stale-server";

function health(overrides: Partial<HealthResponse> = {}): HealthResponse {
  return { version: "0.3.1", spa_bundle: "index-DlQLqQAV.js", ...overrides };
}

describe("currentBundleName", () => {
  it("pulls the filename out of a module URL", () => {
    expect(currentBundleName("http://localhost:8000/assets/index-DlQLqQAV.js")).toBe(
      "index-DlQLqQAV.js",
    );
  });

  it("ignores a cache-busting query string", () => {
    expect(currentBundleName("/assets/index-DlQLqQAV.js?t=123")).toBe(
      "index-DlQLqQAV.js",
    );
  });

  it("returns empty for a dev-server module with no fingerprint", () => {
    expect(currentBundleName("http://localhost:5173/src/main.tsx")).toBe("");
  });
});

describe("isServerStale", () => {
  it("flags a server that booted against an older bundle", () => {
    // The real case: a gui process left running for hours across a rebuild.
    expect(isServerStale(health({ spa_bundle: "index-OLD00000.js" }), "index-NEW11111.js")).toBe(
      true,
    );
  });

  it("stays quiet when they match", () => {
    expect(isServerStale(health(), "index-DlQLqQAV.js")).toBe(false);
  });

  it("stays quiet when the health check has not answered yet", () => {
    expect(isServerStale(null, "index-DlQLqQAV.js")).toBe(false);
  });

  it("stays quiet under the Vite dev server, which has no fingerprint", () => {
    // Crying wolf here would teach the researcher to ignore the banner.
    expect(isServerStale(health(), "")).toBe(false);
  });

  it("stays quiet when the server cannot name its bundle", () => {
    expect(isServerStale(health({ spa_bundle: "" }), "index-DlQLqQAV.js")).toBe(false);
  });
});

describe("detectStaleServer", () => {
  const BUNDLE = "http://localhost:8000/assets/index-NEW11111.js";

  function jsonResponse(body: unknown) {
    return { ok: true, json: async () => body } as unknown as Response;
  }

  it("flags a server that predates the health endpoint entirely", async () => {
    // It answers with index.html through the SPA catch-all, so .json() throws.
    // This is the upgrade that introduces the check, and the case that cost
    // the most time before it existed.
    const fetchFn = async () =>
      ({
        ok: true,
        json: async () => {
          throw new SyntaxError("Unexpected token '<'");
        },
      }) as unknown as Response;

    expect(await detectStaleServer(fetchFn as typeof fetch, BUNDLE)).toBe(true);
  });

  it("flags a server that booted against an older bundle", async () => {
    const fetchFn = async () =>
      jsonResponse({ version: "0.3.1", spa_bundle: "index-OLD00000.js" });

    expect(await detectStaleServer(fetchFn as typeof fetch, BUNDLE)).toBe(true);
  });

  it("stays quiet when the bundles match", async () => {
    const fetchFn = async () =>
      jsonResponse({ version: "0.3.1", spa_bundle: "index-NEW11111.js" });

    expect(await detectStaleServer(fetchFn as typeof fetch, BUNDLE)).toBe(false);
  });

  it("stays quiet under the Vite dev server", async () => {
    const fetchFn = async () => jsonResponse({ version: "x", spa_bundle: "y" });

    expect(
      await detectStaleServer(fetchFn as typeof fetch, "http://localhost:5173/src/main.tsx"),
    ).toBe(false);
  });

  it("stays quiet when the request fails outright", async () => {
    // A server that is down is a different problem with its own symptoms.
    const fetchFn = async () => ({ ok: false }) as unknown as Response;

    expect(await detectStaleServer(fetchFn as typeof fetch, BUNDLE)).toBe(false);
  });
});
