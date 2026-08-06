import { describe, expect, it } from "vitest";

import {
  PARAM_HELP,
  PARAM_TIER,
  hasNonDefaultAdvanced,
  isAdvanced,
  paramHelp,
} from "./param-help";

describe("completude", () => {
  it("explica todo parâmetro que classifica", () => {
    // The complaint was not knowing what the fields do, so a classified field
    // with no explanation is the exact failure this file exists to prevent.
    for (const key of Object.keys(PARAM_TIER)) {
      expect(PARAM_HELP[key], `sem explicação: ${key}`).toBeTruthy();
    }
  });

  it("não deixa explicação vazia passar", () => {
    for (const [key, text] of Object.entries(PARAM_HELP)) {
      expect(text.trim().length, `explicação vazia: ${key}`).toBeGreaterThan(20);
    }
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
    expect(paramHelp("num_workers")).toMatch(/impede o treino/i);
    expect(paramHelp("num_workers")).toMatch(/1455/);
  });

  it("diz qual knob mexer primeiro quando falta VRAM", () => {
    expect(paramHelp("batch_size")).toMatch(/VRAM/i);
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
