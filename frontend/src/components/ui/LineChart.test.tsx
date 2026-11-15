/** The chart as markup: what is drawn before anyone points at it. The snapping and
 *  placement logic is tested in lib/line-chart.test.ts; the drawing is the browser check's. */
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { LanguageContext } from "../../i18n/context";
import { en } from "../../i18n/en";
import { pt } from "../../i18n/pt";
import { LineChart, type LineChartSeries } from "./LineChart";

function render(
  series: LineChartSeries[],
  extra: Partial<Parameters<typeof LineChart>[0]> = {},
  lang: "pt" | "en" = "en",
) {
  return renderToStaticMarkup(
    <LanguageContext.Provider
      value={{
        lang,
        t: lang === "pt" ? pt : en,
        locale: lang === "pt" ? "pt-BR" : "en-US",
        setLang: () => {},
      }}
    >
      <LineChart title="Loss" series={series} formatValue={(v) => v.toFixed(2)} {...extra} />
    </LanguageContext.Provider>,
  );
}

const a: LineChartSeries = {
  key: "a",
  name: "train",
  color: "red",
  points: [1, 2, 3, 4].map((x) => ({ x, y: 1 / x })),
};

describe("LineChart", () => {
  it("draws a 2px round line, a dot on the last point and labelled ticks", () => {
    const html = render([a]);
    expect(html).toContain('stroke-width="2"');
    expect(html).toContain('stroke-linejoin="round"');
    expect(html).toContain('stroke-linecap="round"');
    expect(html).toContain('r="3.5"');
    expect(html).toContain(">1.00<");
    expect(html).toContain(">4<");
    // Straight segments only: no curve commands in any path.
    for (const d of html.matchAll(/ d="([^"]*)"/g)) expect(d[1]).not.toMatch(/[CcSsQqTtAa]/);
  });

  it("is focusable, named and described for the keyboard", () => {
    const html = render([a]);
    expect(html).toContain('aria-label="Loss"');
    expect(html).toContain('tabindex="0"');
    expect(html).toContain("aria-describedby");
    expect(html).toContain('aria-live="polite"');
    expect(html).toContain(en.chart.keyboardHint.replace("'", "&#x27;"));
    expect(render([a], {}, "pt")).toContain(pt.chart.keyboardHint);
  });

  it("shows no tooltip until an epoch is hovered", () => {
    expect(render([a])).not.toContain("data-tooltip");
    expect(render([a])).not.toContain("data-guide");
  });

  it("fills softly under a single series only", () => {
    expect(render([a])).toContain('fill-opacity="0.07"');
    expect(render([a, { ...a, key: "b", name: "val" }])).not.toContain('fill-opacity="0.07"');
  });

  it("is not drawn without a series or without a measured value", () => {
    expect(render([])).toBe("");
    expect(render([{ ...a, points: [] }])).toBe("");
    expect(
      render([
        {
          ...a,
          points: [
            { x: 1, y: null },
            { x: 2, y: Number.NaN },
          ],
        },
      ]),
    ).toBe("");
  });

  it("leaves out a series that was not reported", () => {
    const html = render([a, { key: "b", name: "val", color: "blue", points: [] }]);
    expect(html).toContain('stroke="red"');
    expect(html).not.toContain('stroke="blue"');
  });

  it("draws a single epoch as a dot", () => {
    const html = render([{ ...a, points: [{ x: 3, y: 0.5 }] }]);
    expect(html).toContain('r="3.5"');
    expect(html).toContain(">3<");
  });

  it("marks an epoch with its label, only when it is inside the data", () => {
    const html = render([a], { marker: { x: 2, color: "green", label: "best" } });
    expect(html).toContain('data-best-epoch="2"');
    expect(html).toContain(">best<");
    expect(render([a], { marker: { x: 99, color: "green" } })).not.toContain("data-best-epoch");
  });

  it("labels a level line once", () => {
    const flat = { ...a, points: [1, 2, 3].map((x) => ({ x, y: 1 })) };
    expect(render([flat]).match(/>1\.00</g)).toHaveLength(1);
  });

  it("labels very small values in the caller's format", () => {
    const tiny = {
      ...a,
      points: [
        { x: 1, y: 2.73e-6 },
        { x: 2, y: 5e-6 },
      ],
    };
    const html = render([tiny], { formatValue: (v) => v.toExponential(2) });
    expect(html).toContain("e-6<");
  });

  it("breaks a line at a gap", () => {
    const gap = {
      ...a,
      points: [
        { x: 1, y: 1 },
        { x: 2, y: 2 },
        { x: 3, y: null },
        { x: 4, y: 3 },
        { x: 5, y: 4 },
      ],
    };
    const d = /<path d="([^"]*)" fill="none"/.exec(render([gap]))![1];
    expect(d.match(/M/g)).toHaveLength(2);
  });

  it("keeps the markup small with many epochs", () => {
    const many = { ...a, points: Array.from({ length: 300 }, (_, i) => ({ x: i + 1, y: Math.sin(i / 9) })) };
    expect(render([many]).match(/<text/g)!.length).toBeLessThan(20);
  });
});
