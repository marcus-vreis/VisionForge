import { describe, expect, it } from "vitest";

import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import {
  PARAM_TIER,
  hasNonDefaultAdvanced,
  isAdvanced,
  paramHelp,
} from "./param-help";

// The help texts live in the dictionaries, one per language. `paramHelp` is an
// open record (the keys are the backend's field names), so the compiler cannot
// check coverage there: these tests do, for every language.
const LANGUAGES = { pt, en };

describe("completude", () => {
  for (const [lang, dict] of Object.entries(LANGUAGES)) {
    it(`explica todo parâmetro que classifica (${lang})`, () => {
      // The complaint was not knowing what the fields do, so a classified field
      // with no explanation is the exact failure this file exists to prevent.
      for (const key of Object.keys(PARAM_TIER)) {
        expect(dict.paramHelp[key], `sem explicação em ${lang}: ${key}`).toBeTruthy();
      }
    });

    it(`não deixa explicação vazia passar (${lang})`, () => {
      for (const [key, text] of Object.entries(dict.paramHelp)) {
        expect(text.trim().length, `explicação vazia em ${lang}: ${key}`).toBeGreaterThan(20);
      }
    });
  }

  it("explica os mesmos campos nos dois idiomas", () => {
    expect(Object.keys(en.paramHelp).sort()).toEqual(Object.keys(pt.paramHelp).sort());
  });

  it("devolve a explicação do idioma ativo, ou nada para um campo sem ajuda", () => {
    expect(paramHelp(pt, "epochs")).toBe(pt.paramHelp.epochs);
    expect(paramHelp(en, "epochs")).toBe(en.paramHelp.epochs);
    expect(paramHelp(en, "um_campo_novo")).toBeUndefined();
  });
});

describe("o corte básico/avançado", () => {
  it("mantém visíveis os quatro que mudam entre experimentos", () => {
    for (const key of ["epochs", "batch_size", "learning_rate", "seed"]) {
      expect(isAdvanced(key), key).toBe(false);
    }
  });

  it("colapsa os que se define uma vez e deixa quieto", () => {
    for (const key of ["optimizer", "weight_decay", "num_workers", "pin_memory"]) {
      expect(isAdvanced(key), key).toBe(true);
    }
  });

  it("trata um parâmetro desconhecido como básico em vez de escondê-lo", () => {
    // A field nobody classified must stay on screen, not vanish by accident.
    expect(isAdvanced("um_campo_novo")).toBe(false);
  });
});

describe("advertências específicas", () => {
  it("avisa que num_workers impede o treino, não o deixa lento", () => {
    // ADR-081: this is the one knob whose wrong value stops training outright.
    expect(paramHelp(pt, "num_workers")).toMatch(/impede o treino/i);
    expect(paramHelp(pt, "num_workers")).toMatch(/1455/);
    expect(paramHelp(en, "num_workers")).toMatch(/keeps training from starting/i);
    expect(paramHelp(en, "num_workers")).toMatch(/1455/);
  });

  it("diz qual knob mexer primeiro quando falta VRAM", () => {
    expect(paramHelp(pt, "batch_size")).toMatch(/VRAM/i);
    expect(paramHelp(en, "batch_size")).toMatch(/VRAM/i);
    expect(paramHelp(en, "batch_size")).toMatch(/lower this one first/i);
  });
});

describe("hasNonDefaultAdvanced", () => {
  const defaults = { epochs: 10, optimizer: "adam", weight_decay: 0.0 };

  it("abre a seção quando um valor avançado foi ajustado", () => {
    expect(hasNonDefaultAdvanced({ ...defaults, weight_decay: 0.01 }, defaults)).toBe(
      true,
    );
  });

  it("fica fechada num formulário intocado", () => {
    expect(hasNonDefaultAdvanced({ ...defaults }, defaults)).toBe(false);
  });

  it("ignora mudança em campo básico", () => {
    // Changing epochs is the normal case; it must not force the section open.
    expect(hasNonDefaultAdvanced({ ...defaults, epochs: 50 }, defaults)).toBe(false);
  });

  it("compara por valor, não por referência", () => {
    const d = { freeze: [0, 1] };
    expect(hasNonDefaultAdvanced({ freeze: [0, 1] }, d)).toBe(false);
    expect(hasNonDefaultAdvanced({ freeze: [0, 2] }, d)).toBe(true);
  });
});

describe("quando a seção avançada abre sozinha", () => {
  it("não abre por um default que o schema não declara", () => {
    // The scheduler is a nested object and carries no `default` of its own, so
    // treating "undefined" as different opened the section on every load.
    expect(
      hasNonDefaultAdvanced({ scheduler: { kind: "none" } }, { scheduler: undefined }),
    ).toBe(false);
  });

  it("abre quando um avançado foi realmente alterado", () => {
    expect(hasNonDefaultAdvanced({ optimizer: "sgd" }, { optimizer: "adam" })).toBe(
      true,
    );
  });

  it("ignora mudanças em campos básicos", () => {
    expect(hasNonDefaultAdvanced({ epochs: 50 }, { epochs: 20 })).toBe(false);
  });
});
