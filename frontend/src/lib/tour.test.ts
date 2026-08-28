import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  CARD_WIDTH,
  TOUR_STEPS,
  clearTourSeen,
  markTourSeen,
  placeCard,
  readTourSeen,
} from "./tour";

const KEY = "vf.tour.seen";

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

describe("the seen flag", () => {
  it("starts unset, so the guide is offered on a fresh machine", () => {
    installStorage(memoryStorage());

    expect(readTourSeen()).toBe(false);
  });

  it("stays set after the guide is finished or dismissed", () => {
    installStorage(memoryStorage());

    markTourSeen();

    expect(readTourSeen()).toBe(true);
  });

  it("is forgotten again by clearTourSeen", () => {
    installStorage(memoryStorage({ [KEY]: "1" }));

    clearTourSeen();

    expect(readTourSeen()).toBe(false);
  });

  it("reads as unseen when storage throws, instead of breaking the screen", () => {
    installStorage({
      getItem: () => {
        throw new Error("storage bloqueado");
      },
    });

    expect(readTourSeen()).toBe(false);
  });

  it("swallows a write failure: a preference is not worth a crash", () => {
    installStorage({
      setItem: () => {
        throw new Error("storage bloqueado");
      },
    });

    expect(() => markTourSeen()).not.toThrow();
  });
});

describe("the script", () => {
  it("names an existing anchor or none at all", () => {
    // A typo here would silently centre the card instead of pointing at the
    // element, so the set of anchors is pinned to what the components mark.
    const marked = new Set([
      "tabs",
      "dataset",
      "device",
      "train",
      "history",
      "datasets",
    ]);

    for (const step of TOUR_STEPS) {
      if (step.anchor) expect(marked).toContain(step.anchor);
    }
  });

  it("gives every step something to say", () => {
    for (const step of TOUR_STEPS) {
      expect(step.title.length).toBeGreaterThan(0);
      expect(step.body.length).toBeGreaterThan(40);
    }
  });
});

describe("placeCard", () => {
  const view = { width: 1280, height: 800 };
  const box = (o: Partial<DOMRect>): DOMRect =>
    ({ left: 0, top: 0, width: 0, height: 0, right: 0, bottom: 0, ...o }) as DOMRect;

  it("centres when the step has no target", () => {
    const p = placeCard(null, 254, view);

    expect(p.left).toBe((1280 - CARD_WIDTH) / 2);
    expect(p.top).toBe((800 - 254) / 2);
  });

  it("sits under the target when the space below is enough", () => {
    // Um alvo no alto: 100 + 254 + 20 cabe em 800.
    const p = placeCard(box({ top: 60, bottom: 100, left: 540, right: 740, width: 200 }), 254, view);

    expect(p.top).toBe(116); // 100 + GAP
  });

  it("goes above the target when the bottom has no room", () => {
    // A barra inferior fixa é o caso real: nada cabe embaixo dela.
    const p = placeCard(box({ top: 713, bottom: 764, left: 584, right: 757, width: 173 }), 254, view);

    expect(p.top).toBe(713 - 16 - 254);
  });

  it("centres on the target horizontally", () => {
    const p = placeCard(box({ top: 60, bottom: 100, left: 500, right: 700, width: 200 }), 254, view);

    expect(p.left).toBe(600 - CARD_WIDTH / 2);
  });

  it("stays on screen when the target hangs off the right edge", () => {
    const p = placeCard(box({ top: 60, bottom: 100, left: 1200, right: 1400, width: 200 }), 254, view);

    expect(p.left).toBe(1280 - CARD_WIDTH - 20);
  });

  it("stays on screen when the target is below the fold", () => {
    // Acontece de verdade: a rolagem suave ainda não terminou quando o passo
    // muda, e sem o limite o cartão iria parar fora da tela.
    const p = placeCard(box({ top: 1900, bottom: 2000, left: 100, right: 900, width: 800 }), 254, view);

    expect(p.top).toBeLessThanOrEqual(800 - 254 - 20);
    expect(p.top).toBeGreaterThanOrEqual(20);
  });

  it("never inverts on a screen smaller than the card", () => {
    const p = placeCard(box({ top: 10, bottom: 40, left: 0, right: 300, width: 300 }), 600, {
      width: 320,
      height: 480,
    });

    expect(p.left).toBe(20);
    expect(p.top).toBe(20);
  });
});
