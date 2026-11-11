import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import { freshSeeds, type RerunSource } from "../lib/history-rerun";
import { ExperimentNameCard } from "./ExperimentHeader";
import { ReplicatesCard } from "./ReplicatesCard";
import { RerunActions } from "./RerunActions";

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };
const BUILT_INS = ["classification", "detection", "regression", "segmentation", "anomaly"];
const handle = { families: BUILT_INS, onRerun: () => {} };
const r = pt.runDetail.rerun;

function render(node: ReactNode): string {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={lang}>{node}</LanguageContext.Provider>,
  );
}

/** The `<button …>` tags of the markup, one string each. */
function buttons(html: string): string[] {
  return html.match(/<button[^>]*>[^<]*<\/button>/g) ?? [];
}

describe("RerunActions (#200)", () => {
  it("offers both actions on a run whose task has a tab", () => {
    const run: RerunSource = { run_id: "r1", config: { name: "x", training: { seed: 42 } } };
    const html = render(<RerunActions source={run} handle={handle} />);

    const [load, repeat] = buttons(html);
    expect(load).toContain(r.load);
    expect(load).not.toContain("disabled");
    expect(repeat).toContain(r.replicates(3));
    expect(repeat).toContain(r.replicatesTitle("43, 44, 45"));
    expect(html).not.toContain('role="note"');
  });

  it("shows both off, with the reason on screen, for a removed researcher's task", () => {
    const run: RerunSource = {
      run_id: "r2",
      task: "custom:leaf_counter",
      config: { name: "leaf", training: { seed: 1 } },
    };
    const html = render(<RerunActions source={run} handle={handle} />);

    const all = buttons(html);
    expect(all).toHaveLength(2);
    for (const button of all) expect(button).toContain("disabled");
    // Written beside the buttons, not only in a tooltip.
    expect(html).toContain(`role="note"`);
    expect(html).toContain(">" + r.customRemoved("leaf_counter").replace(/"/g, "&quot;") + "<");
  });

  it("offers no repeat on a sweep", () => {
    const sweep: RerunSource = {
      run_id: "s1",
      config: { name: "x", block: "grid_search" },
      group: { kind: "sweep", seeds: [] },
    };
    const all = buttons(render(<RerunActions source={sweep} handle={handle} />));
    expect(all).toHaveLength(1);
    expect(all[0]).toContain(r.load);
  });
});

describe("the cards a History import fills (#200)", () => {
  it("opens Réplicas on the new seeds, three of them", () => {
    const seeds = freshSeeds([42]);
    const html = render(
      <ReplicatesCard
        metrics={[{ value: "r2", label: "R²" }]}
        accent="#fff"
        onReplicates={() => {}}
        preset={{ id: 1, seeds }}
      />,
    );

    expect(html).toContain('value="43, 44, 45"');
    expect(html).not.toContain('value="42, 43, 44, 45, 46"');
  });

  it("shows the import's warning where the YAML button shows it", () => {
    const html = render(
      <ExperimentNameCard
        name="exp"
        onNameChange={() => {}}
        placeholder="exp"
        onExportYaml={() => {}}
        onImportConfig={() => null}
        importNotice={{ id: 1, source: "20261101_exp", problem: "training › epochs: inteiro" }}
      />,
    );
    expect(html).toContain("training › epochs: inteiro");
    expect(html).not.toContain(pt.experimentHeader.imported("20261101_exp"));
  });

  it("names the run when it went through clean", () => {
    const html = render(
      <ExperimentNameCard
        name="exp"
        onNameChange={() => {}}
        placeholder="exp"
        onExportYaml={() => {}}
        onImportConfig={() => null}
        importNotice={{ id: 1, source: "20261101_exp", problem: null }}
      />,
    );
    expect(html).toContain(pt.experimentHeader.imported("20261101_exp"));
  });
});
