import { describe, expect, it } from "vitest";

import { countUnits, unitState } from "./unit-status";

describe("unitState", () => {
  it("reads the three statuses the server records", () => {
    expect(unitState("success")).toBe("ok");
    expect(unitState("stopped")).toBe("stopped");
    expect(unitState("failed")).toBe("failed");
  });

  it("does not take a status it has never seen for a success", () => {
    expect(unitState("")).toBe("failed");
    expect(unitState(undefined)).toBe("failed");
    expect(unitState(null)).toBe("failed");
    expect(unitState("brand_new")).toBe("failed");
  });
});

describe("countUnits", () => {
  const rows = [
    { status: "success" },
    { status: "success" },
    { status: "stopped" },
    { status: "failed" },
  ];

  it("keeps a stopped unit apart from a failed one", () => {
    expect(countUnits(rows, "ok")).toBe(2);
    expect(countUnits(rows, "stopped")).toBe(1);
    expect(countUnits(rows, "failed")).toBe(1);
  });
});
