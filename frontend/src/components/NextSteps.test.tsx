import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { en } from "../i18n/en";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import { nextStepItems, type NextStep } from "../lib/next-steps";
import { NextSteps } from "./NextSteps";

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };

function render(node: ReactNode): string {
  return renderToStaticMarkup(<LanguageContext.Provider value={lang}>{node}</LanguageContext.Provider>);
}

function buttons(html: string): string[] {
  return html.match(/<button[^>]*>.*?<\/button>/g) ?? [];
}

const STILL_IMPROVING: NextStep = { id: "stillImproving", action: "reuse" };
const UNTESTED: NextStep = { id: "untested", action: "test" };

describe("NextSteps", () => {
  it("draws nothing when no suggestion holds", () => {
    expect(render(<NextSteps steps={[]} handlers={{}} />)).toBe("");
  });

  it("draws a titled list with one sentence and one real button per suggestion", () => {
    const html = render(
      <NextSteps steps={[STILL_IMPROVING, UNTESTED]} handlers={{ reuse: () => {}, test: () => {} }} />,
    );

    expect(html).toContain(pt.nextSteps.title);
    expect(html.match(/<li/g)).toHaveLength(2);
    expect(html).toMatch(/<section aria-labelledby="([^"]+)".*<h3 id="\1"/);
    expect(html).toContain("<ul");
    expect(html).toContain(pt.nextSteps.stillImproving);
    expect(html).toContain(pt.nextSteps.untested);
    const [reuse, test] = buttons(html);
    expect(reuse).toContain('type="button"');
    expect(reuse).toContain(pt.runDetail.rerun.load);
    expect(test).toContain(pt.runDetail.tests.open);
  });

  it("keeps the sentence and drops the button when the panel has no handler", () => {
    const html = render(<NextSteps steps={[{ id: "singleRun", action: "replicate" }]} handlers={{}} />);
    expect(html).toContain(pt.nextSteps.singleRun);
    expect(buttons(html)).toEqual([]);
  });

  it("disables a button the panel says is busy", () => {
    const html = render(
      <NextSteps
        steps={[{ id: "unfinished", action: "resume" }]}
        handlers={{ resume: () => {} }}
        disabled={{ resume: true }}
      />,
    );
    expect(buttons(html)[0]).toContain("disabled");
  });
});

describe("nextStepItems", () => {
  it("wires each button to the panel's handler for its action", () => {
    const replicate = vi.fn();
    const [item] = nextStepItems(pt, [{ id: "singleRun", action: "replicate" }], { replicate });
    expect(item.action?.label).toBe(pt.runDetail.rerun.replicates(3));
    item.action?.onClick();
    expect(replicate).toHaveBeenCalledOnce();
  });

  it("asks to resume only when resuming is offered", () => {
    const step: NextStep = { id: "unfinished", action: "resume" };
    const [withResume] = nextStepItems(pt, [step], { resume: () => {} });
    const [without] = nextStepItems(pt, [step], {});
    expect(withResume.text).toBe(pt.nextSteps.unfinishedResume);
    expect(withResume.action?.label).toBe(pt.nextSteps.resume);
    expect(without.text).toBe(pt.nextSteps.unfinished);
    expect(without.action).toBeNull();
  });

  it("speaks the reader's language", () => {
    const [item] = nextStepItems(en, [UNTESTED], { test: () => {} });
    expect(item.text).toBe(en.nextSteps.untested);
    expect(item.action?.label).toBe(en.runDetail.tests.open);
  });
});
