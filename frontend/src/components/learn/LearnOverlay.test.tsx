/**
 * The test setup has no DOM, so the page is rendered to markup and read as
 * text. Scrolling, the rail following the reader and Esc are left to the
 * browser check.
 */
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { LanguageContext } from "../../i18n/context";
import { en } from "../../i18n/en";
import { pt } from "../../i18n/pt";
import { LEARN_CHAPTERS, PROJECT_DOCS, RESOURCES, STACK, chapterAnchor } from "../../lib/learn";
import { GuideMenu } from "../GuideMenu";
import { CurvesFigure } from "./LearnFigures";
import { LearnOverlay } from "./LearnOverlay";

const inLanguage = (lang: "pt" | "en", node: React.ReactNode) =>
  renderToStaticMarkup(
    <LanguageContext.Provider
      value={{
        lang,
        t: lang === "pt" ? pt : en,
        locale: lang === "pt" ? "pt-BR" : "en-US",
        setLang: () => {},
      }}
    >
      {node}
    </LanguageContext.Provider>,
  );

const page = (lang: "pt" | "en") =>
  inLanguage(lang, <LearnOverlay onClose={() => {}} onFirstTraining={() => {}} />);

/** React escapes quotes and ampersands in text; the dictionaries do not. */
const escaped = (text: string) =>
  text.replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("'", "&#x27;");

describe("LearnOverlay", () => {
  it("is a labelled dialog with the title and the lead", () => {
    for (const lang of ["pt", "en"] as const) {
      const dict = lang === "pt" ? pt : en;
      const html = page(lang);
      expect(html).toContain('role="dialog"');
      expect(html).toContain('aria-labelledby="learn-title"');
      expect(html).toContain(`<h1 id="learn-title">${escaped(dict.learn.title)}</h1>`);
      expect(html).toContain(escaped(dict.learn.lead));
    }
  });

  it("shows every chapter once, in order, each with its anchor", () => {
    for (const lang of ["pt", "en"] as const) {
      const dict = lang === "pt" ? pt : en;
      const html = page(lang);
      const positions = LEARN_CHAPTERS.map((id) => {
        const heading = `<h2>${escaped(dict.learn.chapters[id].title)}</h2>`;
        expect(html.split(heading)).toHaveLength(2);
        expect(html).toContain(`id="${chapterAnchor(id)}"`);
        return html.indexOf(heading);
      });
      expect(positions).toEqual([...positions].sort((a, b) => a - b));
    }
  });

  it("lists every chapter in the rail and marks the first as current", () => {
    const html = page("pt");
    const rail = html.slice(html.indexOf("<nav"), html.indexOf("</nav>"));
    for (const id of LEARN_CHAPTERS) expect(rail).toContain(escaped(pt.learn.chapters[id].title));
    expect(rail.split('aria-current="true"')).toHaveLength(2);
    expect(rail.indexOf('aria-current="true"')).toBeLessThan(
      rail.indexOf(escaped(pt.learn.chapters[LEARN_CHAPTERS[1]].title)),
    );
  });

  it("opens every outside link in a new tab without an opener", () => {
    const html = page("pt");
    const links = html.match(/<a [^>]*>/g) ?? [];
    expect(links).toHaveLength(STACK.length + RESOURCES.length + PROJECT_DOCS.length);
    for (const link of links) {
      expect(link).toContain('target="_blank"');
      expect(link).toContain('rel="noopener noreferrer"');
    }
    for (const item of [...STACK, ...RESOURCES, ...PROJECT_DOCS]) {
      expect(html).toContain(`href="${escaped(item.url)}"`);
    }
  });

  it("gives every image a description and its size", () => {
    const html = page("en");
    const images = html.match(/<img [^>]*>/g) ?? [];
    expect(images).toHaveLength(3);
    for (const image of images) {
      expect(image).toMatch(/alt="[^"]{10,}"/);
      expect(image).toMatch(/width="\d+"/);
      expect(image).toMatch(/height="\d+"/);
    }
  });

  it("says which figures are illustrative", () => {
    // The split, the curves and the matrix; the drawing and the screens are not.
    expect(page("pt").split(escaped(pt.learn.illustrative))).toHaveLength(4);
  });

  it("ends the recording chapter with the way into the first training", () => {
    const html = page("pt");
    expect(html).toContain(escaped(pt.learn.chapters.watch.start));
  });
});

describe("CurvesFigure", () => {
  it("explains the rate it opens on and marks its best epoch only at the right rate", () => {
    const good = inLanguage("pt", <CurvesFigure />);
    expect(good).toContain("overfitting");
    expect(good).toMatch(/melhor época: \d+/);
    expect(good.split('aria-pressed="true"')).toHaveLength(2);

    const low = inLanguage("pt", <CurvesFigure initial="low" />);
    expect(low).toContain(escaped(pt.learn.chapters.curves.notes.low));
    expect(low).not.toMatch(/melhor época: \d+/);
  });
});

describe("GuideMenu with the page", () => {
  it("lists it first when it can be opened, and not at all otherwise", () => {
    const withPage = inLanguage(
      "pt",
      <GuideMenu onSelect={() => {}} onLearn={() => {}} initialOpen />,
    );
    const at = withPage.indexOf(pt.learn.menuTitle);
    expect(at).toBeGreaterThan(0);
    expect(at).toBeLessThan(withPage.indexOf(pt.guides.tour.title));
    expect(withPage).toContain(escaped(pt.learn.menuSummary));

    const without = inLanguage("pt", <GuideMenu onSelect={() => {}} initialOpen />);
    expect(without).not.toContain(pt.learn.menuTitle);
  });
});
