import { describe, expect, it } from "vitest";

import { displayName, greetingFor, shouldGreet } from "./greeting";

function memoryStorage(seed: Record<string, string> = {}): Storage {
  const map = new Map(Object.entries(seed));
  return {
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => void map.set(k, v),
    removeItem: (k: string) => void map.delete(k),
    clear: () => map.clear(),
    key: () => null,
    length: 0,
  } as unknown as Storage;
}

describe("displayName", () => {
  it("capitalises a plain account name", () => {
    expect(displayName("marcu")).toBe("Marcu");
  });

  it("drops a Windows domain prefix", () => {
    expect(displayName("LAB\\marcus")).toBe("Marcus");
  });

  it("keeps only the first token of a dotted name", () => {
    expect(displayName("marcus.vinicius")).toBe("Marcus");
  });

  it("handles an email-shaped account", () => {
    expect(displayName("marcus@lab.edu")).toBe("Marcus");
  });

  it("preserves accents rather than mangling them", () => {
    expect(displayName("ítalo")).toBe("Ítalo");
  });

  it("returns nothing for a name with no letters", () => {
    // "Bem-vindo, 12345" reads like a database record, not a hello.
    expect(displayName("12345")).toBe("");
    expect(displayName("__")).toBe("");
  });

  it("returns nothing for absent input", () => {
    expect(displayName(null)).toBe("");
    expect(displayName(undefined)).toBe("");
    expect(displayName("")).toBe("");
  });
});

describe("greetingFor", () => {
  it("names the researcher when it can", () => {
    expect(greetingFor("marcu")).toBe("Bem-vindo, Marcu");
  });

  it("greets plainly rather than awkwardly when it cannot", () => {
    expect(greetingFor(null)).toBe("Bem-vindo");
    expect(greetingFor("12345")).toBe("Bem-vindo");
  });
});

describe("shouldGreet", () => {
  it("greets on the first call of a session", () => {
    expect(shouldGreet(memoryStorage())).toBe(true);
  });

  it("stays quiet on a reload mid-work", () => {
    // A reload is not an arrival; replaying the animation would be a flicker.
    const storage = memoryStorage();

    expect(shouldGreet(storage)).toBe(true);
    expect(shouldGreet(storage)).toBe(false);
  });

  it("greets when storage is unavailable rather than staying silent", () => {
    expect(shouldGreet(undefined)).toBe(true);
  });

  it("greets when storage throws", () => {
    const hostile = {
      getItem: () => {
        throw new Error("blocked");
      },
    } as unknown as Storage;

    expect(shouldGreet(hostile)).toBe(true);
  });
});
