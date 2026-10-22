import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, auditDataset, downloadRunBundle } from "./client";

afterEach(() => vi.unstubAllGlobals());

function storage() {
  vi.stubGlobal("localStorage", { getItem: (key: string) => ({ "vf.profile": "ana", "vf.lang": "en" })[key as "vf.profile"] ?? null });
}

describe("image audit request", () => {
  it("preserves custom split folders, CSV and strict mode with profile headers", async () => {
    storage();
    const report = { ok: false, duplicate_groups: [{ paths: ["fit/a.png", "holdout/a.png"] }] };
    const fetcher = vi.fn(async () => new Response(JSON.stringify(report)));
    vi.stubGlobal("fetch", fetcher);
    const body = { base_dir: "C:/data", splits: { train: "fit", val: "holdout", test: "final" }, group_manifest: "groups.csv", strict_groups: true };
    expect(await auditDataset(body)).toEqual(report);
    const [url, init] = fetcher.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/dataset/audit");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual(body);
    expect(new Headers(init.headers).get("X-VF-Profile")).toBe("ana");
    expect(new Headers(init.headers).get("X-VF-Lang")).toBe("en");
    expect(new Headers(init.headers).get("Content-Type")).toBe("application/json");
  });

  it("propagates a validation response rather than reporting a clean audit", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "Missing dataset" }), { status: 400 })));
    await expect(auditDataset({ base_dir: "missing" })).rejects.toMatchObject({ status: 400, message: "Missing dataset" });
  });
});

describe("experiment bundle download", () => {
  it("downloads the real response blob and server filename using the active profile", async () => {
    storage();
    const blob = new Blob(["ZIP evidence"], { type: "application/zip" });
    const fetcher = vi.fn(async () => new Response(blob, { headers: { "Content-Disposition": 'attachment; filename="evidence.zip"' } }));
    vi.stubGlobal("fetch", fetcher);
    const anchor = { href: "", download: "", click: vi.fn(), remove: vi.fn() };
    const appendChild = vi.fn();
    vi.stubGlobal("document", { createElement: () => anchor, body: { appendChild } });
    const createObjectURL = vi.fn(() => "blob:evidence");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { createObjectURL, revokeObjectURL });

    await downloadRunBundle("run /1");

    const [url, init] = fetcher.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/runs/run%20%2F1/bundle");
    expect(new Headers(init.headers).get("X-VF-Profile")).toBe("ana");
    expect(new Headers(init.headers).get("X-VF-Lang")).toBe("en");
    expect(createObjectURL.mock.calls[0][0]).toBeInstanceOf(Blob);
    expect(await (createObjectURL.mock.calls[0][0] as Blob).text()).toBe("ZIP evidence");
    expect(anchor).toMatchObject({ href: "blob:evidence", download: "evidence.zip" });
    expect(appendChild).toHaveBeenCalledWith(anchor);
    expect(anchor.click).toHaveBeenCalledOnce();
    expect(anchor.remove).toHaveBeenCalledOnce();
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:evidence");
  });

  it.each([400, 404, 409])("surfaces HTTP %s and the server's reason without creating a download", async (status) => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: "Run unavailable" }), { status })));
    await expect(downloadRunBundle("r")).rejects.toMatchObject({ status, message: "Run unavailable" });
  });

  it("handles non-JSON and network failures", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("bad gateway", { status: 502 })));
    await expect(downloadRunBundle("r")).rejects.toMatchObject({ status: 502, message: "HTTP 502" });
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("offline"); }));
    await expect(downloadRunBundle("r")).rejects.toBeInstanceOf(ApiError);
    await expect(downloadRunBundle("r")).rejects.toMatchObject({ status: 0, cause: "offline" });
  });
});
