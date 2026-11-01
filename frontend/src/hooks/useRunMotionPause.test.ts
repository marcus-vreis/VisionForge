import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { markRunActive } from "./useRunMotionPause";

describe("markRunActive", () => {
  it("sets the mark and clears it on cleanup", () => {
    const root = { dataset: {} as Record<string, string | undefined> };
    const clear = markRunActive(root);
    expect(root.dataset.runActive).toBe("true");
    clear();
    expect(root.dataset).not.toHaveProperty("runActive");
  });

  it("has a stylesheet rule that ends infinite animations under the mark", () => {
    const css = readFileSync(resolve(__dirname, "../index.css"), "utf-8");
    expect(css).toMatch(
      /html\[data-run-active\][^{]*\{\s*animation-iteration-count:\s*1 !important;/,
    );
  });
});
