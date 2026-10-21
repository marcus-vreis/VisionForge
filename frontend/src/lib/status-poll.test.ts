import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../api/client";
import { POLL_MS, startStatusPoll, type ConnectionState } from "./status-poll";

/** What `request()` throws when fetch itself fails: the server never answered. */
const offline = () => new ApiError(0, "Não foi possível conectar ao servidor.");

/** A poll whose ticks fail for the first `down` calls, then succeed. */
function watch(down: number, answer: () => Error = offline) {
  let calls = 0;
  const connection: (ConnectionState | null)[] = [];
  const errors: unknown[] = [];
  const stop = startStatusPoll({
    tick: async () => {
      calls += 1;
      if (calls <= down) throw answer();
    },
    onConnection: (c) => connection.push(c),
    onError: (e) => errors.push(e),
  });
  return { stop, connection, errors, calls: () => calls };
}

describe("startStatusPoll", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("keeps watching through a few seconds without the server, then says it is back", async () => {
    const w = watch(2);

    await vi.advanceTimersByTimeAsync(POLL_MS);
    expect(w.connection).toEqual(["reconnecting"]);
    expect(w.errors).toEqual([]);

    await vi.advanceTimersByTimeAsync(20_000);
    expect(w.calls()).toBeGreaterThan(3);
    expect(w.connection).toEqual(["reconnecting", "reconnecting", null]);
    expect(w.errors).toEqual([]);
    w.stop();
  });

  it("backs off between retries instead of hammering every 2 s", async () => {
    const w = watch(Infinity);
    await vi.advanceTimersByTimeAsync(POLL_MS); // first failure
    await vi.advanceTimersByTimeAsync(2_000); // second, after 2 s
    expect(w.calls()).toBe(2);
    await vi.advanceTimersByTimeAsync(3_999);
    expect(w.calls()).toBe(2); // the third waits 4 s
    await vi.advanceTimersByTimeAsync(1);
    expect(w.calls()).toBe(3);
    w.stop();
  });

  it("declares the connection lost after ~30 s, keeps trying, and resumes when the server answers", async () => {
    const w = watch(7);
    await vi.advanceTimersByTimeAsync(POLL_MS + 29_999);
    expect(w.connection.at(-1)).toBe("reconnecting");
    await vi.advanceTimersByTimeAsync(1);
    expect(w.connection.at(-1)).toBe("lost");
    expect(w.errors).toEqual([]);

    await vi.advanceTimersByTimeAsync(30_000);
    expect(w.connection.at(-1)).toBeNull();
    expect(w.errors).toEqual([]);
    w.stop();
  });

  it("ends the watch on an error the server answered", async () => {
    const http = new ApiError(500, "boom");
    const w = watch(1, () => http);
    await vi.advanceTimersByTimeAsync(POLL_MS);
    expect(w.errors).toEqual([http]);
    await vi.advanceTimersByTimeAsync(20_000);
    expect(w.calls()).toBe(1);
  });

  it("does not poll again once stopped, even from a tick in flight", async () => {
    let calls = 0;
    const stop = startStatusPoll({
      tick: async () => {
        calls += 1;
        stop();
        throw offline();
      },
      onConnection: () => {},
      onError: () => {},
    });
    await vi.advanceTimersByTimeAsync(60_000);
    expect(calls).toBe(1);
  });
});
