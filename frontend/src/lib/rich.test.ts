import { describe, expect, it } from "vitest";

import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { parseRich, type RichKind, type RichSegment } from "./rich";

const plain = (text: string): RichSegment[] => [{ kind: "text", text }];

describe("parseRich", () => {
  it("devolve texto sem marcas como um único trecho", () => {
    expect(parseRich("só texto")).toEqual(plain("só texto"));
    expect(parseRich("")).toEqual([]);
  });

  it("reconhece code, strong e em", () => {
    expect(parseRich("a `b` c **d** e __f__ g")).toEqual([
      { kind: "text", text: "a " },
      { kind: "code", text: "b" },
      { kind: "text", text: " c " },
      { kind: "strong", text: "d" },
      { kind: "text", text: " e " },
      { kind: "em", text: "f" },
      { kind: "text", text: " g" },
    ]);
  });

  it("aceita marcas no começo, no fim e coladas umas nas outras", () => {
    expect(parseRich("`a`, fim `b`").map((s) => s.kind)).toEqual(["code", "text", "code"]);
    expect(parseRich("**s**__e__`c`").map((s) => s.kind)).toEqual(["strong", "em", "code"]);
    expect(parseRich("`a``b`").map((s) => s.text)).toEqual(["a", "b"]);
  });

  // The bug these guard against: a plain segment that begins with a delimiter
  // was treated as if it were a mark, and lost its first and last character.
  it.each([
    "`sem fechar no começo",
    "meio ` solto",
    "texto e `só abre",
    "**strong sem fechar",
    "__init_ no começo",
    "**a*b** asterisco dentro",
    "__foo_bar__ sublinhado dentro",
    "``",
    "****",
    "____",
  ])("mostra delimitador sem par como texto: %s", (input) => {
    expect(parseRich(input)).toEqual(plain(input));
  });

  it("não aninha: o que está dentro de uma marca aparece como foi escrito", () => {
    expect(parseRich("**`nested`**")).toEqual([{ kind: "strong", text: "`nested`" }]);
  });

  it("deixa HTML como texto", () => {
    expect(parseRich("<b>x</b> `y`")).toEqual([
      { kind: "text", text: "<b>x</b> " },
      { kind: "code", text: "y" },
    ]);
  });

  it("não perde nem inventa caracteres de texto", () => {
    const input = "a `b` **c** __d__ e ` f";
    const rebuilt = parseRich(input)
      .map((s) => ({ text: s.text, kind: s.kind }))
      .map(({ kind, text }) =>
        kind === "code" ? `\`${text}\`` : kind === "strong" ? `**${text}**` : kind === "em" ? `__${text}__` : text,
      )
      .join("");
    expect(rebuilt).toBe(input);
  });
});

type Tree = { [key: string]: unknown };

/** The sentences components render through <Rich>, whether or not they carry a mark today. */
const RICH_PATHS = [
  "paramPanel.blockHints.crossValidation",
  "paramPanel.blockHints.transferLearning",
  "paramPanel.blockHints.gridSearch",
  "paramPanel.blockHints.randomSearch",
  "paramPanel.grid.banner",
  "paramPanel.randomSearch.example",
];

function at(tree: Tree, path: string): unknown {
  return path.split(".").reduce<unknown>((node, key) => (node as Tree | undefined)?.[key], tree);
}

/** Every string in a dictionary that carries a mark, by path: by convention a
 * Rich sentence, so a mark added to any text is held to the same rules. */
function markedStrings(tree: Tree, prefix = ""): Array<[string, string]> {
  return Object.entries(tree).flatMap(([key, value]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    if (typeof value === "string") return /`|\*\*|__/.test(value) ? [[path, value] as [string, string]] : [];
    if (typeof value === "object" && value !== null) return markedStrings(value as Tree, path);
    return [];
  });
}

const count = (text: string, kind: RichKind) => parseRich(text).filter((s) => s.kind === kind).length;

describe("as frases com marcas dos dicionários", () => {
  const collect = (dict: Tree) => {
    const found = new Map(markedStrings(dict));
    for (const path of RICH_PATHS) {
      const text = at(dict, path);
      expect(typeof text, path).toBe("string");
      found.set(path, text as string);
    }
    return found;
  };
  const ptRich = collect(pt);
  const enRich = collect(en);

  it("estão nos dois idiomas", () => {
    expect([...enRich.keys()].sort()).toEqual([...ptRich.keys()].sort());
  });

  for (const [lang, rich] of [["pt", ptRich], ["en", enRich]] as const) {
    it(`não deixam delimitador sobrando depois de interpretar (${lang})`, () => {
      for (const [path, text] of rich) {
        const leftover = parseRich(text)
          .filter((s) => s.kind === "text")
          .map((s) => s.text)
          .join("")
          .match(/`|\*\*|__/g);
        expect(leftover, `${lang} ${path}`).toBeNull();
      }
    });
  }

  it("têm as mesmas marcas em pt e en", () => {
    for (const [path, ptText] of ptRich) {
      const enText = enRich.get(path) ?? "";
      for (const kind of ["code", "strong", "em"] as const) {
        expect(count(enText, kind), `${path} (${kind})`).toBe(count(ptText, kind));
      }
    }
  });
});
