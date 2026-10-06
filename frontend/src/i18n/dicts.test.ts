import { describe, expect, it } from "vitest";
import { en } from "./en";
import { pt } from "./pt";

type Tree = { [key: string]: unknown };

function leaves(tree: Tree, prefix = ""): Array<[string, unknown]> {
  return Object.entries(tree).flatMap(([key, value]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return typeof value === "object" && value !== null
      ? leaves(value as Tree, path)
      : [[path, value] as [string, unknown]];
  });
}

describe("dictionaries", () => {
  const ptLeaves = new Map(leaves(pt));
  const enLeaves = new Map(leaves(en));

  it("have the same keys", () => {
    expect([...enLeaves.keys()].sort()).toEqual([...ptLeaves.keys()].sort());
  });

  it("have no empty text", () => {
    for (const [path, value] of [...ptLeaves, ...enLeaves]) {
      if (typeof value === "string") expect(value.trim(), path).not.toBe("");
    }
  });

  it("agree on which entries take values", () => {
    for (const [path, value] of ptLeaves) {
      expect(typeof enLeaves.get(path), path).toBe(typeof value);
      if (typeof value === "function") {
        expect((enLeaves.get(path) as (...a: unknown[]) => string).length, path).toBe(
          (value as (...a: unknown[]) => string).length,
        );
      }
    }
  });

  it("word a count with the plural right in both languages", () => {
    expect(pt.detectionDatasetStats.applied(1)).toBe("1 classe aplicada");
    expect(pt.detectionDatasetStats.applied(3)).toBe("3 classes aplicadas");
    expect(en.detectionDatasetStats.applied(1)).toBe("1 class applied");
    expect(en.detectionDatasetStats.applied(3)).toBe("3 classes applied");

    expect(pt.paramPanel.fieldErrors(1)).toBe("1 campo com erro:");
    expect(pt.paramPanel.fieldErrors(3)).toBe("3 campos com erro:");
    expect(en.paramPanel.fieldErrors(1)).toBe("1 field with errors:");
    expect(en.paramPanel.fieldErrors(3)).toBe("3 fields with errors:");

    const warnings = (dict: typeof pt, n: number) =>
      dict.paramPanel.importWarnings(n, "x", 0).split(":")[0];
    expect(warnings(pt, 1)).toBe("YAML importado com 1 aviso estrutural");
    expect(warnings(pt, 2)).toBe("YAML importado com 2 avisos estruturais");
    expect(warnings(en, 1)).toBe("YAML imported with 1 structural warning");
    expect(warnings(en, 2)).toBe("YAML imported with 2 structural warnings");

    expect(pt.runDetail.gradcam.done(1, "layer4")).toBe("1 mapa gerado · camada layer4");
    expect(pt.runDetail.gradcam.done(3, "layer4")).toBe("3 mapas gerados · camada layer4");
    expect(en.runDetail.gradcam.done(1, "layer4")).toBe("1 map generated · target layer: layer4");
    expect(en.runDetail.gradcam.done(3, "layer4")).toBe("3 maps generated · target layer: layer4");
  });
});
