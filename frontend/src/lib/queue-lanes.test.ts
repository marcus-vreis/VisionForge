import { describe, expect, it } from "vitest";

import type { QueuedJobInfo, QueueSnapshot } from "../types/run";
import {
  colabLane,
  findQueuedJob,
  isColabLane,
  laneAccount,
  queueLanes,
  queueTotals,
  withAccountLane,
} from "./queue-lanes";

const job = (run_id: string, lane = "local"): QueuedJobInfo => ({
  run_id,
  label: run_id,
  task: "classification",
  strategy: "simple",
  submitted_at: "2026-11-05T00:00:00Z",
  lane,
});

const COLAB = "colab:ana@gmail.com";

// A local job and a Colab job running at once, each with one waiting behind it.
const both: QueueSnapshot = {
  active: job("local-1"),
  pending: [job("local-2")],
  lanes: [
    { lane: COLAB, active: job("colab-1", COLAB), pending: [job("colab-2", COLAB)] },
    { lane: "local", active: job("local-1"), pending: [job("local-2")] },
  ],
};

describe("the server's queues (ADR-163 phase 4)", () => {
  it("lists the local queue first, whatever order the server sends", () => {
    expect(queueLanes(both).map((l) => l.lane)).toEqual(["local", COLAB]);
  });

  it("reads a server without lanes as one local queue", () => {
    const old: QueueSnapshot = { active: job("a"), pending: [job("b")] };
    expect(queueLanes(old)).toEqual([{ lane: "local", active: job("a"), pending: [job("b")] }]);
  });

  it("counts the waiting jobs of every queue, not only the local one", () => {
    expect(queueTotals(both)).toEqual({ pending: 2, running: true });
    const onlyColab: QueueSnapshot = {
      active: null,
      pending: [],
      lanes: [
        { lane: "local", active: null, pending: [] },
        { lane: COLAB, active: job("colab-1", COLAB), pending: [] },
      ],
    };
    // The local queue is idle, but the server is not: the Colab job runs.
    expect(queueTotals(onlyColab)).toEqual({ pending: 0, running: true });
  });

  it("finds a job in its own queue, with its place in that queue's line", () => {
    expect(findQueuedJob(both, "colab-2")).toMatchObject({
      lane: COLAB,
      running: false,
      position: 1,
      waiting: 1,
    });
    expect(findQueuedJob(both, "colab-1")).toMatchObject({ lane: COLAB, running: true });
    expect(findQueuedJob(both, "local-2")).toMatchObject({ lane: "local", position: 1 });
    expect(findQueuedJob(both, "gone")).toBeNull();
  });

  it("names the account of a Colab queue, and the profile when the CLI could not", () => {
    expect(laneAccount(COLAB)).toBe("ana@gmail.com");
    expect(laneAccount("colab:profile:lab")).toBe("lab");
    expect(laneAccount("local")).toBeNull();
    expect(isColabLane(COLAB)).toBe(true);
    expect(isColabLane("local")).toBe(false);
    expect(isColabLane(undefined)).toBe(false);
  });

  it("shows the connected account's queue before its first job, keyed as the server keys it", () => {
    const local = [{ lane: "local", active: null, pending: [] }];
    expect(colabLane("Ana@Gmail.com")).toBe(COLAB);
    expect(withAccountLane(local, "Ana@Gmail.com").map((l) => l.lane)).toEqual(["local", COLAB]);
    // Not twice once the server has it.
    expect(withAccountLane(queueLanes(both), "ana@gmail.com")).toHaveLength(2);
    expect(withAccountLane(local, null)).toBe(local);
  });
});
