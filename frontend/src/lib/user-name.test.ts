import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  clearUserName,
  normalizeUserName,
  readUserName,
  saveUserName,
} from "./user-name";

const KEY = "vf.welcome.name";

function installStorage(impl: Partial<Storage>): void {
  vi.stubGlobal("localStorage", impl as Storage);
}

function memoryStorage(seed: Record<string, string> = {}): Storage {
  const map = new Map(Object.entries(seed));
  return {
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => void map.set(k, v),
    removeItem: (k: string) => void map.delete(k),
  } as unknown as Storage;
}

beforeEach(() => {
  vi.unstubAllGlobals();
});

describe("normalizeUserName", () => {
  it("trims and collapses whitespace", () => {
    expect(normalizeUserName("  Marcus   Vinicius  ")).toBe("Marcus Vinicius");
  });

  it("caps at 32 characters, which is what the header chip fits", () => {
    expect(normalizeUserName("a".repeat(50))).toHaveLength(32);
  });

  it("leaves an empty entry empty, so the form can refuse it", () => {
    expect(normalizeUserName("   ")).toBe("");
  });

  it("preserves the case that was typed", () => {
    // The chip shows the name verbatim; capitalising it for them would be rude.
    expect(normalizeUserName("marcus")).toBe("marcus");
    expect(normalizeUserName("MARCUS")).toBe("MARCUS");
  });
});

describe("read / save / clear", () => {
  it("round-trips a name", () => {
    installStorage(memoryStorage());

    saveUserName("Marcus");

    expect(readUserName()).toBe("Marcus");
  });

  it("reads empty when nothing was saved", () => {
    installStorage(memoryStorage());

    expect(readUserName()).toBe("");
  });

  it("trims a stored value that arrived padded", () => {
    installStorage(memoryStorage({ [KEY]: "  Marcus  " }));

    expect(readUserName()).toBe("Marcus");
  });

  it("clearing brings the intro back", () => {
    installStorage(memoryStorage({ [KEY]: "Marcus" }));

    clearUserName();

    expect(readUserName()).toBe("");
  });
});

describe("blocked storage", () => {
  it("reads empty rather than throwing in private mode", () => {
    // Without a saved name the intro simply runs again; nothing breaks.
    installStorage({
      getItem: () => {
        throw new Error("blocked");
      },
    });

    expect(readUserName()).toBe("");
  });

  it("saving is a no-op rather than an error", () => {
    installStorage({
      setItem: () => {
        throw new Error("blocked");
      },
    });

    expect(() => saveUserName("Marcus")).not.toThrow();
  });

  it("clearing is a no-op rather than an error", () => {
    installStorage({
      removeItem: () => {
        throw new Error("blocked");
      },
    });

    expect(() => clearUserName()).not.toThrow();
  });
});
