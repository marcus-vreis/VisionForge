import { describe, expect, it } from "vitest";

import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { greeting } from "./greeting";

describe("greeting (ADR-126)", () => {
  it("follows the pronoun in Portuguese", () => {
    expect(greeting(pt, "ele")).toBe("Bem-vindo");
    expect(greeting(pt, "ela")).toBe("Bem-vinda");
    expect(greeting(pt, "neutro")).toBe("Boas-vindas");
    expect(greeting(pt, "ela", "Luana")).toBe("Bem-vinda, Luana");
  });

  it("greets in the neutral form while no pronoun was chosen", () => {
    expect(greeting(pt, null)).toBe("Boas-vindas");
    expect(greeting(pt, undefined, "Marcus")).toBe("Boas-vindas, Marcus");
  });

  it("says Welcome for everyone in English", () => {
    for (const p of ["ele", "ela", "neutro", null] as const) {
      expect(greeting(en, p)).toBe("Welcome");
    }
  });
});
