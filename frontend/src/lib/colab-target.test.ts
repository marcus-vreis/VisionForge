import { describe, expect, it } from "vitest";

import {
  colabAccountFacts,
  colabBlock,
  colabDataProblem,
  colabLaunchOf,
  defaultColabSetup,
} from "./colab-target";

const drive = defaultColabSetup("G:\\Meu Drive");

describe("defaultColabSetup", () => {
  it("starts on a T4 and on the Drive, with nothing picked", () => {
    expect(defaultColabSetup()).toEqual({
      accelerator: "T4",
      kind: "drive",
      link: "",
      driveRoot: "",
      drivePath: "",
      driveRemote: null,
      driveExport: false,
    });
  });
});

describe("colabDataProblem", () => {
  it("accepts a picked Drive folder whatever the form holds", () => {
    expect(colabDataProblem({ ...drive, drivePath: "datasets/beans" }, "")).toBeNull();
  });

  it("falls back to the form's folder when no Drive folder was picked", () => {
    expect(colabDataProblem(drive, "G:\\Meu Drive\\beans")).toBeNull();
  });

  it("has nothing to send with neither a picked folder nor a form folder", () => {
    expect(colabDataProblem(drive, "  ")).toBe("no_folder");
  });

  it("refuses a link the VM cannot download", () => {
    const link = { ...drive, kind: "link" as const };
    expect(colabDataProblem({ ...link, link: "C:\\datasets\\beans" }, "x")).toBe("bad_link");
    expect(colabDataProblem({ ...link, link: "" }, "x")).toBe("bad_link");
    expect(colabDataProblem({ ...link, link: "https://a.b/d.zip" }, "")).toBeNull();
    expect(colabDataProblem({ ...link, link: "torchvision:CIFAR10" }, "")).toBeNull();
  });
});

describe("colabLaunchOf", () => {
  it("sends the picked Drive folder, the GPU and the export switch", () => {
    const setup = {
      ...drive,
      accelerator: "L4" as const,
      drivePath: "datasets/beans",
      driveExport: true,
    };
    expect(colabLaunchOf(setup, "")).toEqual({
      source: { kind: "drive", my_drive_root: "G:\\Meu Drive", path: "datasets/beans" },
      accelerator: "L4",
      drive_export: true,
    });
  });

  it("leaves the Drive path out when the form's own folder is the dataset", () => {
    expect(colabLaunchOf({ ...drive, driveRoot: "" }, "G:\\Meu Drive\\beans")).toEqual({
      source: { kind: "drive" },
      accelerator: "T4",
      drive_export: false,
    });
  });

  it("sends a public dataset by its provider", () => {
    const launch = colabLaunchOf({ ...drive, kind: "link", link: "hf:beans/beans" }, "");
    expect(launch?.source).toEqual({
      kind: "provider",
      provider: "huggingface",
      dataset: "beans/beans",
    });
  });

  it("is null while the origin cannot be sent", () => {
    expect(colabLaunchOf(drive, "")).toBeNull();
    expect(colabLaunchOf({ ...drive, kind: "link", link: "beans" }, "x")).toBeNull();
  });
});

describe("colabBlock", () => {
  it("lets a plain classification through", () => {
    expect(colabBlock({ task: "classification", custom: false, strategy: "simple" })).toBeNull();
  });

  it("refuses the other tasks and a researcher's own", () => {
    expect(colabBlock({ task: "detection", custom: false, strategy: "simple" })).toBe("task");
    expect(colabBlock({ task: "classification", custom: true, strategy: "simple" })).toBe("task");
  });

  it("refuses the strategies that are several runs", () => {
    expect(colabBlock({ task: "classification", custom: false, strategy: "replicates" })).toBe(
      "replicates",
    );
    expect(
      colabBlock({ task: "classification", custom: false, strategy: "replicatedComparison" }),
    ).toBe("comparison");
  });
});

describe("colabAccountFacts", () => {
  const usage = { balance: 200, rate_per_hour: null, active_assignments: 0, hours_left: null };

  it("reads the balance and counts the runtimes alive, with their hardware once", () => {
    const sessions = [
      { name: "a", hardware: "T4", ours: true },
      { name: "b", hardware: "T4", ours: false },
      { name: "c", hardware: "L4", ours: true },
    ];
    expect(colabAccountFacts(usage, sessions)).toEqual({
      balance: "200",
      runtimes: 3,
      allocated: ["T4", "L4"],
    });
  });

  it("keeps one decimal of a small balance", () => {
    expect(colabAccountFacts({ ...usage, balance: 12.34 }, []).balance).toBe("12.3");
  });

  it("has no balance before the account answered or when it reports none", () => {
    expect(colabAccountFacts(null, null)).toEqual({ balance: null, runtimes: 0, allocated: [] });
    expect(colabAccountFacts({ ...usage, balance: null }, []).balance).toBeNull();
  });
});
