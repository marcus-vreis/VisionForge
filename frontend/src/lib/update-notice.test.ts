import { describe, expect, it } from "vitest";

import type { UpdateStatus } from "../api/client";
import { noticeMode, waitForVersion } from "./update-notice";

const status = (over: Partial<UpdateStatus> = {}): UpdateStatus => ({
  current: "0.14.0",
  latest: "0.15.0",
  available: true,
  can_apply: true,
  blocked: null,
  command: "pip install --upgrade visionforge-studio==0.15.0",
  last_attempt: null,
  check_disabled: false,
  ...over,
});

describe("noticeMode", () => {
  it("offers the button when the server can install", () => {
    expect(noticeMode(status(), null)).toEqual({ kind: "offer", latest: "0.15.0" });
  });

  it("says nothing without a newer release, offline, or before the answer", () => {
    expect(noticeMode(status({ available: false }), null).kind).toBe("hidden");
    expect(noticeMode(status({ latest: null, available: false }), null).kind).toBe("hidden");
    expect(noticeMode(null, null).kind).toBe("hidden");
  });

  it("stays quiet for a skipped version, and comes back for the next one", () => {
    expect(noticeMode(status(), "0.15.0").kind).toBe("hidden");
    expect(noticeMode(status({ latest: "0.16.0" }), "0.15.0").kind).toBe("offer");
  });

  it("shows the reason and the command when the button cannot install", () => {
    expect(
      noticeMode(status({ can_apply: false, blocked: "source", command: "git pull" }), null),
    ).toEqual({ kind: "manual", latest: "0.15.0", reason: "source", command: "git pull" });
  });

  it("puts a failed attempt at the version on offer first", () => {
    const failed = status({
      last_attempt: {
        target: "0.15.0",
        ok: false,
        returncode: 1,
        finished_at: "2026-10-15T12:00:00Z",
        log: "C:/vf/update.log",
      },
    });
    expect(noticeMode(failed, null)).toEqual({
      kind: "failed",
      target: "0.15.0",
      log: "C:/vf/update.log",
    });
  });

  it("forgets a failure at an older target once a newer release is on offer", () => {
    const old = status({
      latest: "0.16.0",
      last_attempt: { target: "0.15.0", ok: false, returncode: 1, finished_at: "", log: "" },
    });
    expect(noticeMode(old, null).kind).toBe("offer");
  });
});

describe("waitForVersion", () => {
  /** A fake clock and server: `answers` are what successive polls get. */
  const run = (answers: Array<string | null>, timeoutMs = 10_000) => {
    let clock = 0;
    let i = 0;
    return waitForVersion("0.15.0", async () => answers[Math.min(i++, answers.length - 1)], {
      timeoutMs,
      intervalMs: 1_000,
      now: () => clock,
      sleep: async (ms) => {
        clock += ms;
      },
    });
  };

  it("reports the update once the new version answers", async () => {
    await expect(run(["0.14.0", null, null, "0.15.0"])).resolves.toBe("updated");
  });

  it("stops early when the old version comes back after going down", async () => {
    await expect(run([null, null, "0.14.0"])).resolves.toBe("back_old");
  });

  it("keeps waiting while the old server has not gone down yet", async () => {
    await expect(run(["0.14.0", "0.14.0", null, "0.15.0"])).resolves.toBe("updated");
  });

  it("gives up at the deadline", async () => {
    await expect(run([null], 5_000)).resolves.toBe("timeout");
  });
});
