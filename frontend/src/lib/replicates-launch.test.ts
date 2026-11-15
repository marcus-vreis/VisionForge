import { afterEach, describe, expect, it, vi } from "vitest";
import { replicatesLaunch } from "./replicates-launch";

/** A fetch that records its calls and answers like a queued run. */
function stubFetch() {
  const fetchMock = vi.fn<(url: string, init?: RequestInit) => Promise<unknown>>(
    async () => ({
      ok: true,
      status: 200,
      json: async () => ({ run_id: "r1", status: "running" }),
    }),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => vi.unstubAllGlobals());

const classificationForm = {
  name: "exp",
  block: "classification",
  model: { name: "resnet18", num_classes: 3 },
  device: { kind: "cpu", gpu_ids: [] },
};

describe("replicatesLaunch (ADR-056)", () => {
  it("sends classification replicates to the classification route with the seeds", async () => {
    const fetchMock = stubFetch();
    const { body, run } = replicatesLaunch("classification", classificationForm, {
      metric: "accuracy",
      seeds: [42, 43, 44],
    });

    await run(body);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/classification\/replicates$/);
    const sent = JSON.parse(String(init?.body)) as Record<string, unknown>;
    expect(sent["seeds"]).toEqual([42, 43, 44]);
    expect(sent["metric"]).toBe("accuracy");
    expect((sent["config"] as Record<string, unknown>)["block"]).toBe("classification");
  });

  it("keeps the classification block even when the form still names another", () => {
    const { body } = replicatesLaunch(
      "classification",
      { ...classificationForm, block: "transfer_learning" },
      { metric: "accuracy", n_replicates: 5 },
    );

    expect((body["config"] as Record<string, unknown>)["block"]).toBe("classification");
    expect(body["n_replicates"]).toBe(5);
  });

  it("leaves a standalone task's config as it came", () => {
    const config = { name: "reg", model: { name: "resnet18" } };
    const { body } = replicatesLaunch("regression", config, { metric: "r2", n_replicates: 3 });

    expect(body["config"]).toEqual(config);
  });
});
