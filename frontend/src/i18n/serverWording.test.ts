import { describe, expect, it } from "vitest";

import { en } from "./en";
import { pt } from "./pt";

/**
 * Texts the page shows on a local install and on a shared server alike
 * (ADR-212). On a server the browser is not where keys and datasets live, so
 * these must not say "this computer" or "a local folder".
 */
describe("wording that holds on a shared server too", () => {
  const local = /neste computador|pasta local|this computer|local folder|datasets\//i;

  it.each([
    ["pt", pt],
    ["en", en],
  ] as const)("%s", (_lang, dict) => {
    const texts = [
      dict.wandb.tokenHint,
      dict.comet.tokenHint,
      dict.sampleDataset.title,
      dict.datasetsOverlay.title,
      dict.datasetsOverlay.description,
    ];
    for (const text of texts) expect(text).not.toMatch(local);
  });

  it("never sends a download to 'a local folder'", () => {
    // Downloads land in the person's datasets folder, on the server when there is one.
    expect(JSON.stringify(pt)).not.toMatch(/pasta local/i);
    expect(JSON.stringify(en)).not.toMatch(/local folder/i);
  });

  it("names the datasets sheet for what all its tabs hold", () => {
    expect(pt.datasetsOverlay.title).toBe("Dados de treino");
    expect(en.datasetsOverlay.title).toBe("Training data");
  });
});
