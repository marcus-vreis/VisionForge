import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import type { ClassSimilarityReport, SimilarityPair } from "../lib/class-similarity";
import { ClassSimilarityView } from "./ClassSimilarity";

function render(lang: "pt" | "en", report: ClassSimilarityReport) {
  const html = renderToStaticMarkup(
    <LanguageContext.Provider value={{ lang, t: lang === "pt" ? pt : en, locale: lang, setLang: () => {} }}>
      <ClassSimilarityView report={report} root="/data/ds" />
    </LanguageContext.Provider>,
  );
  // Compare with the text as written, not its HTML escapes.
  return html.replaceAll("&#x27;", "'");
}

const pair = (over: Partial<SimilarityPair> = {}): SimilarityPair => ({
  a: "cat",
  b: "dog",
  a_to_b: 1,
  b_to_a: 1 / 3,
  centroid_similarity: 0.5162,
  flagged: true,
  examples: [
    { path: "train/cat/c2.png", class: "cat", neighbour: "val/dog/d3.png", neighbour_class: "dog", similarity: 0.9988 },
  ],
  ...over,
});

const report = (pairs: SimilarityPair[]): ClassSimilarityReport => ({
  model: "resnet18 (ImageNet)",
  per_class: 200,
  images_per_class: { cat: 2, dog: 3, fox: 2 },
  sampled: false,
  cancelled: false,
  unreadable: 0,
  described: 7,
  classes: [
    { name: "cat", used: 2, purity: 0 },
    { name: "dog", used: 3, purity: 2 / 3 },
    { name: "fox", used: 2, purity: 1 },
  ],
  flag_share: 0.1,
  flagged_count: pairs.filter((p) => p.flagged).length,
  pair_count: pairs.length,
  pairs,
});

describe("the similar classes report (ADR-239)", () => {
  it.each(["pt", "en"] as const)("flags the pair and shows where the classes meet in %s", (lang) => {
    const t = (lang === "pt" ? pt : en).classSimilarity;
    const html = render(lang, report([pair()]));
    expect(html).toContain(t.found(1, "10%"));
    expect(html).toContain("100%");
    expect(html).toContain("33%");
    expect(html).toContain("0.52");
    expect(html).toContain("(0.999)");
    expect(html).toContain(t.examplesTitle("cat", "dog"));
    expect(html).toContain("path=val%2Fdog%2Fd3.png");
    expect(html).toContain(t.nearest("dog", 0.9988));
    expect(html).toContain("dog: 67% (3)");
  });

  it("lists an unflagged pair with its images too: one misfiled image is a small share", () => {
    const html = render("en", report([pair({ a_to_b: 0.05, b_to_a: 0, flagged: false })]));
    expect(html).toContain(en.classSimilarity.none("10%"));
    expect(html).toContain("5%");
    expect(html).toContain(en.classSimilarity.examplesTitle("cat", "dog"));
  });

  it("counts the pairs past the ones listed", () => {
    const r = { ...report([pair()]), pair_count: 31 };
    expect(render("pt", r)).toContain(pt.classSimilarity.morePairs(30));
  });
});
