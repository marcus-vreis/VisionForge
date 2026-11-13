import { describe, expect, it } from "vitest";
import { childRel, driveCrumbs, driveLaunchSource } from "./drive-folders";

describe("driveCrumbs", () => {
  it("is empty at the root", () => {
    expect(driveCrumbs("")).toEqual([]);
  });

  it("gives each folder down the path the rel that lists it", () => {
    expect(driveCrumbs("datasets/flores/train")).toEqual([
      { name: "datasets", rel: "datasets" },
      { name: "flores", rel: "datasets/flores" },
      { name: "train", rel: "datasets/flores/train" },
    ]);
  });

  it("ignores stray separators", () => {
    expect(driveCrumbs("/a//b/")).toEqual([
      { name: "a", rel: "a" },
      { name: "b", rel: "a/b" },
    ]);
  });
});

describe("childRel", () => {
  it("joins a sub-folder under the root and under a folder", () => {
    expect(childRel("", "datasets")).toBe("datasets");
    expect(childRel("datasets", "flores")).toBe("datasets/flores");
  });
});

describe("driveLaunchSource", () => {
  it("keeps the old request when nothing was typed or picked", () => {
    expect(driveLaunchSource("", "  ")).toEqual({ kind: "drive" });
    expect(driveLaunchSource(" G:\\Meu Drive ", "")).toEqual({
      kind: "drive",
      my_drive_root: "G:\\Meu Drive",
    });
  });

  it("sends the folder picked in the Drive browser", () => {
    expect(driveLaunchSource("", "G:\\Meu Drive\\datasets\\flores ")).toEqual({
      kind: "drive",
      path: "G:\\Meu Drive\\datasets\\flores",
    });
  });
});
