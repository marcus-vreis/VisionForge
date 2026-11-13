import { describe, expect, it } from "vitest";

import { runStatusLabel } from "./run-status-color";

describe("runStatusLabel", () => {
  const words = { completed: "concluído", failed: "falhou" };

  it("uses the dictionary's word for a status it knows", () => {
    expect(runStatusLabel("completed", words)).toBe("concluído");
  });

  it("shows a status the dictionary does not know as the server wrote it", () => {
    expect(runStatusLabel("queued", words)).toBe("queued");
  });

  it("does not read a word off the object's prototype", () => {
    expect(runStatusLabel("toString", words)).toBe("toString");
  });
});
