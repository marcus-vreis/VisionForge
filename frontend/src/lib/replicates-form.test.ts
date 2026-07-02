import { describe, expect, it } from "vitest";
import {
  buildReplicatesPayload,
  parseSeeds,
  seedsProblem,
} from "./replicates-form";

describe("replicates-form", () => {
  it("parses comma-separated seeds, dropping junk and flooring floats", () => {
    expect(parseSeeds("42, 43,  44 ")).toEqual([42, 43, 44]);
    expect(parseSeeds("1, abc, 2.9, -5, ,3")).toEqual([1, 2, 3]);
    expect(parseSeeds("")).toEqual([]);
  });

  it("flags fewer than two seeds and duplicates (mirrors the backend 422s)", () => {
    expect(seedsProblem([42])).toMatch(/pelo menos 2/);
    expect(seedsProblem([1, 1, 2])).toMatch(/duplicadas/);
    expect(seedsProblem([1, 2])).toBeNull();
  });

  it("auto mode sends n_replicates and omits seeds", () => {
    const p = buildReplicatesPayload("auto", 5, "7, 8", "accuracy");
    expect(p).toEqual({ metric: "accuracy", n_replicates: 5 });
  });

  it("explicit mode sends the parsed seed list and omits n_replicates", () => {
    const p = buildReplicatesPayload("explicit", 5, "7, 8, 9", "r2");
    expect(p).toEqual({ metric: "r2", seeds: [7, 8, 9] });
  });
});
