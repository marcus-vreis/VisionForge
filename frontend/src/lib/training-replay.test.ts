import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { canReplay, createReplay, REPLAY_STEPS } from "./training-replay";
import type { StoryStage } from "./training-story";

describe("canReplay", () => {
  it("is only for a finished run, and not for PatchCore", () => {
    expect(canReplay("done", "cnn")).toBe(true);
    expect(canReplay("done", "patchcore")).toBe(false);
    for (const stage of ["queued", "preparing", "training", "testing", "interrupted", "failed"] as const) {
      expect(canReplay(stage, "cnn"), stage).toBe(false);
    }
  });
});

describe("createReplay", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  const total = REPLAY_STEPS.reduce((sum, s) => sum + s.ms, 0);

  it("lasts 10 to 15 s", () => {
    expect(total).toBeGreaterThanOrEqual(10_000);
    expect(total).toBeLessThanOrEqual(15_000);
  });

  it("walks the stages in order and returns to the real one", () => {
    const seen: Array<StoryStage | null> = [];
    createReplay((s) => seen.push(s)).start();
    expect(seen).toEqual(["preparing"]);
    vi.advanceTimersByTime(total - 1);
    expect(seen).toEqual(["preparing", "training", "testing"]);
    vi.advanceTimersByTime(1);
    expect(seen).toEqual(["preparing", "training", "testing", null]);
    vi.advanceTimersByTime(total);
    expect(seen).toHaveLength(4);
  });

  it("can be played again after it ends", () => {
    const seen: Array<StoryStage | null> = [];
    const replay = createReplay((s) => seen.push(s));
    replay.start();
    vi.advanceTimersByTime(total);
    replay.start();
    expect(seen.at(-1)).toBe("preparing");
    vi.advanceTimersByTime(total);
    expect(seen.at(-1)).toBeNull();
  });

  it("restarting mid-way begins again without a second clock", () => {
    const seen: Array<StoryStage | null> = [];
    const replay = createReplay((s) => seen.push(s));
    replay.start();
    vi.advanceTimersByTime(3500);
    replay.start();
    expect(seen.at(-1)).toBe("preparing");
    vi.advanceTimersByTime(total);
    expect(seen.filter((s) => s === null)).toHaveLength(1);
  });

  it("cancel stops the clock and shows the real stage", () => {
    const seen: Array<StoryStage | null> = [];
    const replay = createReplay((s) => seen.push(s));
    replay.start();
    replay.cancel();
    vi.advanceTimersByTime(total);
    expect(seen).toEqual(["preparing", null]);
  });
});
