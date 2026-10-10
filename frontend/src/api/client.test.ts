import { afterEach, describe, expect, it, vi } from "vitest";

import {
  createProfile,
  deleteRun,
  downloadRunMarkdown,
  fetchProfiles,
  fetchRuns,
  withLang,
  withProfile,
} from "./client";

/** A fetch that records its calls and answers with `body`. */
function stubFetch(body: unknown = [], status = 200) {
  const fetchMock = vi.fn(async () => ({
    ok: status < 400,
    status,
    json: async () => body,
  }));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function stubStorage(initial: Record<string, string>) {
  const data = new Map(Object.entries(initial));
  vi.stubGlobal("localStorage", {
    getItem: (k: string) => data.get(k) ?? null,
    setItem: (k: string, v: string) => void data.set(k, v),
    removeItem: (k: string) => void data.delete(k),
  });
}

/** The header named `name` on the init of call `n`, however it was given. */
function sentHeader(fetchMock: ReturnType<typeof stubFetch>, name: string, n = 0) {
  const calls = fetchMock.mock.calls as unknown as [string, RequestInit | undefined][];
  const init = calls[n][1];
  return new Headers(init?.headers).get(name);
}

afterEach(() => vi.unstubAllGlobals());

describe("withProfile", () => {
  it("adds the profile header to an init that had none", () => {
    const init = withProfile(undefined, "ana");

    expect(new Headers(init.headers).get("X-VF-Profile")).toBe("ana");
  });

  it("keeps the headers and the rest of the init it was given", () => {
    const original = {
      method: "POST",
      body: "{}",
      headers: { "Content-Type": "application/json" },
    };

    const init = withProfile(original, "ana");

    expect(init.method).toBe("POST");
    expect(init.body).toBe("{}");
    const headers = new Headers(init.headers);
    expect(headers.get("Content-Type")).toBe("application/json");
    expect(headers.get("X-VF-Profile")).toBe("ana");
    // And the caller's own object is not rewritten.
    expect(original.headers).toEqual({ "Content-Type": "application/json" });
  });

  it("sends nothing for the default profile or a malformed slug", () => {
    const init = { method: "DELETE" };

    for (const slug of ["default", "", "../x", "Ana"]) {
      expect(withProfile(init, slug)).toBe(init);
    }
    expect(withProfile(undefined, "default")).toEqual({});
  });
});

describe("every request carries the chosen profile", () => {
  it("reads the choice from the browser's storage on each call", async () => {
    stubStorage({ "vf.profile": "ana" });
    const fetchMock = stubFetch([]);

    await fetchRuns();

    expect(fetchMock).toHaveBeenCalledWith("/api/runs", expect.anything());
    expect(sentHeader(fetchMock, "X-VF-Profile")).toBe("ana");
  });

  it("follows a switch made between two calls", async () => {
    stubStorage({ "vf.profile": "ana" });
    const fetchMock = stubFetch([]);
    await fetchRuns();

    stubStorage({ "vf.profile": "bob" });
    await fetchRuns();

    expect(sentHeader(fetchMock, "X-VF-Profile", 0)).toBe("ana");
    expect(sentHeader(fetchMock, "X-VF-Profile", 1)).toBe("bob");
  });

  it("sends no header for the default profile or with nothing chosen", async () => {
    stubStorage({ "vf.profile": "default" });
    const fetchMock = stubFetch([]);
    await fetchRuns();
    stubStorage({});
    await fetchRuns();

    expect(sentHeader(fetchMock, "X-VF-Profile", 0)).toBeNull();
    expect(sentHeader(fetchMock, "X-VF-Profile", 1)).toBeNull();
  });

  it("does not let a damaged stored value reach the server", async () => {
    stubStorage({ "vf.profile": "../../etc" });
    const fetchMock = stubFetch([]);

    await fetchRuns();

    expect(sentHeader(fetchMock, "X-VF-Profile")).toBeNull();
  });

  it("is added to writes without losing their own headers", async () => {
    stubStorage({ "vf.profile": "ana" });
    const fetchMock = stubFetch({ slug: "bia", name: "Bia", is_default: false });

    await createProfile("Bia");

    expect(sentHeader(fetchMock, "X-VF-Profile")).toBe("ana");
    expect(sentHeader(fetchMock, "Content-Type")).toBe("application/json");
    const calls = fetchMock.mock.calls as unknown as [string, RequestInit][];
    expect(calls[0][1].method).toBe("POST");
    expect(calls[0][1].body).toBe(JSON.stringify({ name: "Bia" }));
  });

  it("is added to a delete", async () => {
    stubStorage({ "vf.profile": "ana" });
    const fetchMock = stubFetch({ run_id: "r", status: "deleted" });

    await deleteRun("r");

    expect(sentHeader(fetchMock, "X-VF-Profile")).toBe("ana");
  });
});

describe("withLang", () => {
  it("adds the language header to an init that had none", () => {
    const init = withLang(undefined, "en");

    expect(new Headers(init.headers).get("X-VF-Lang")).toBe("en");
  });

  it("keeps the headers and the rest of the init it was given", () => {
    const original = {
      method: "POST",
      body: "{}",
      headers: { "Content-Type": "application/json" },
    };

    const init = withLang(original, "pt");

    expect(init.method).toBe("POST");
    expect(init.body).toBe("{}");
    const headers = new Headers(init.headers);
    expect(headers.get("Content-Type")).toBe("application/json");
    expect(headers.get("X-VF-Lang")).toBe("pt");
    expect(original.headers).toEqual({ "Content-Type": "application/json" });
  });
});

describe("every request carries the language on screen", () => {
  it("sends the language the header toggle stored", async () => {
    stubStorage({ "vf.lang": "en" });
    const fetchMock = stubFetch([]);

    await fetchRuns();

    expect(sentHeader(fetchMock, "X-VF-Lang")).toBe("en");
  });

  it("follows a switch made between two calls", async () => {
    stubStorage({ "vf.lang": "en" });
    const fetchMock = stubFetch([]);
    await fetchRuns();

    stubStorage({ "vf.lang": "pt" });
    await fetchRuns();

    expect(sentHeader(fetchMock, "X-VF-Lang", 0)).toBe("en");
    expect(sentHeader(fetchMock, "X-VF-Lang", 1)).toBe("pt");
  });

  it("takes the browser's language when nothing was chosen yet", async () => {
    stubStorage({});
    vi.stubGlobal("navigator", { language: "pt-BR" });
    const fetchMock = stubFetch([]);
    await fetchRuns();

    vi.stubGlobal("navigator", { language: "de-DE" });
    await fetchRuns();

    expect(sentHeader(fetchMock, "X-VF-Lang", 0)).toBe("pt");
    expect(sentHeader(fetchMock, "X-VF-Lang", 1)).toBe("en");
  });

  it("ignores a stored value that is not a language", async () => {
    stubStorage({ "vf.lang": "klingon" });
    vi.stubGlobal("navigator", { language: "pt-BR" });
    const fetchMock = stubFetch([]);

    await fetchRuns();

    expect(sentHeader(fetchMock, "X-VF-Lang")).toBe("pt");
  });

  it("goes together with the profile and with a write's own headers", async () => {
    stubStorage({ "vf.lang": "en", "vf.profile": "ana" });
    const fetchMock = stubFetch({ slug: "bia", name: "Bia", is_default: false });

    await createProfile("Bia");

    expect(sentHeader(fetchMock, "X-VF-Lang")).toBe("en");
    expect(sentHeader(fetchMock, "X-VF-Profile")).toBe("ana");
    expect(sentHeader(fetchMock, "Content-Type")).toBe("application/json");
  });

  it("is also on the download, which does not go through request()", async () => {
    stubStorage({ "vf.lang": "en" });
    const fetchMock = stubFetch(null, 500);

    await expect(downloadRunMarkdown("r")).rejects.toThrow();

    expect(sentHeader(fetchMock, "X-VF-Lang")).toBe("en");
  });
});

describe("fetchProfiles", () => {
  it("unwraps the list", async () => {
    stubFetch({
      profiles: [
        { slug: "default", name: "default", is_default: true },
        { slug: "ana", name: "Ana", is_default: false },
      ],
    });

    expect((await fetchProfiles()).map((p) => p.slug)).toEqual(["default", "ana"]);
  });
});
