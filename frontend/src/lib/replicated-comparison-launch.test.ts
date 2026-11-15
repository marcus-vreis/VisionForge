import { afterEach, describe, expect, it, vi } from "vitest";
import { runCustomReplicatedComparison } from "../api/client";
import { pt } from "../i18n/pt";
import { buildReplicatedComparisonPayload } from "./replicated-comparison-form";
import { comparisonUnits, replicatedComparisonLaunch } from "./replicated-comparison-launch";

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

const regressionConfig = {
  name: "reg",
  model: { name: "resnet18" },
  training: { learning_rate: 0.001 },
  device: { kind: "cpu", gpu_ids: [] },
};

/** The payload the card hands App for lr 0.0001 and 0.01 over 6 seeds. */
function cardPayload() {
  const built = buildReplicatedComparisonPayload(pt, {
    path: "training.learning_rate",
    rawValues: "0.0001, 0.01",
    seedMode: "auto",
    nReplicates: 6,
    rawSeeds: "",
    metric: "r2",
  });
  if (!("payload" in built)) throw new Error(built.problem);
  return built.payload;
}

describe("replicatedComparisonLaunch (ADR-193)", () => {
  it("sends Regressão's comparison to /api/regression/replicated-comparison", async () => {
    const fetchMock = stubFetch();
    const { body, run } = replicatedComparisonLaunch("regression", regressionConfig, cardPayload());

    await run(body);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/regression/replicated-comparison");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({
      config: regressionConfig,
      variants: {
        baseline: {},
        "learning_rate=0.0001": { "training.learning_rate": 0.0001 },
        "learning_rate=0.01": { "training.learning_rate": 0.01 },
      },
      metric: "r2",
      n_replicates: 6,
    });
  });

  it("counts every variant over every seed for the queue banner", () => {
    const { units } = replicatedComparisonLaunch("regression", regressionConfig, cardPayload());

    expect(units).toBe(18);
    expect(comparisonUnits({ variants: { a: {}, b: {} }, metric: "m", seeds: [1, 2, 3] })).toBe(6);
  });

  it("trains classification's variants with the classification block", () => {
    const { body } = replicatedComparisonLaunch(
      "classification",
      { name: "exp", block: "grid_search" },
      cardPayload(),
    );

    expect((body["config"] as Record<string, unknown>)["block"]).toBe("classification");
  });

  it("sends a custom task's comparison to its own route", async () => {
    const fetchMock = stubFetch();

    await runCustomReplicatedComparison("my_task", { config: {}, ...cardPayload() });

    expect(fetchMock.mock.calls[0][0]).toBe("/api/custom/my_task/replicated-comparison");
  });
});
