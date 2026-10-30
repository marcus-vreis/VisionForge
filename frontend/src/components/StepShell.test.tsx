import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import { StepNavContext, countChanged, type StepId } from "../lib/steps";
import { BottomBar } from "./BottomBar";
import { StepShell, type StepDef } from "./StepShell";

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };

function render(steps: StepDef[], active: StepId = "data") {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={lang}>
      <StepNavContext.Provider value={{ stepFor: () => active, setStep: () => {} }}>
        <StepShell taskKey="detection" steps={steps} />
      </StepNavContext.Provider>
    </LanguageContext.Provider>,
  );
}

const steps: StepDef[] = [
  { id: "data", missing: true, content: <div data-tour="dataset">dados</div> },
  { id: "strategy", changes: 0, content: <div data-probe="sweep">sweep card</div> },
  { id: "training", changes: 2, errors: 1, content: <div>treino</div> },
  { id: "review", content: <div>revisão</div> },
];

describe("StepShell", () => {
  it("keeps every step mounted and hides all but the active one", () => {
    const html = render(steps, "strategy");
    // The sweep card listens to the run signal: it must exist off screen too.
    expect(html).toContain('data-probe="sweep"');
    expect(html).toContain('data-tour="dataset"');
    expect(html).toMatch(/<section data-step="data" hidden=""/);
    expect(html).toMatch(/<section data-step="strategy" class=/);
  });

  it("shows each step's state in the rail", () => {
    const html = render(steps);
    expect(html).toContain(">falta<");
    expect(html).toContain(">padrão<");
    // Errors win over changes.
    expect(html).toContain(">1 erro<");
    expect(html).not.toContain("2 ajustes");
  });

  it("offers the review as the next step from the one before it", () => {
    expect(render(steps, "training")).toContain("Revisar →");
  });

  it("falls back to the first step when the remembered one does not exist", () => {
    const html = render(steps, "image");
    expect(html).toContain('data-active-step="data"');
  });
});

describe("countChanged", () => {
  it("counts dotted paths that differ from the defaults", () => {
    const defaults = { training: { epochs: 50, lr0: 0.01 }, data: { source: "folder" } };
    const current = { training: { epochs: 80, lr0: 0.01 }, data: { source: "yaml" } };
    expect(countChanged(current, defaults, ["training.epochs", "training.lr0", "data.source"])).toBe(2);
  });
});

describe("BottomBar train lock", () => {
  const bar = (lockReason: string | null, isRunning = false) =>
    renderToStaticMarkup(
      <LanguageContext.Provider value={lang}>
        <BottomBar
          onHistory={() => {}}
          onDatasets={() => {}}
          onQueue={() => {}}
          onTrain={() => {}}
          disabled={isRunning}
          historyCount={0}
          queuedCount={0}
          serverRunning={false}
          selection={{ kind: "cpu", gpu_ids: null }}
          onSelectionChange={() => {}}
          isRunning={isRunning}
          lockReason={lockReason}
        />
      </LanguageContext.Provider>,
    );
  const trainButton = (html: string) => html.match(/<button[^>]*data-tour="train"[^>]*>/)?.[0] ?? "";

  it("locks Treinar with the reason as its tooltip", () => {
    const button = trainButton(bar(pt.bottomBar.trainLockedReview));
    expect(button).toContain("disabled");
    expect(button).toContain(`title="${pt.bottomBar.trainLockedReview}"`);
  });

  it("is enabled when nothing locks it", () => {
    expect(trainButton(bar(null))).not.toContain("disabled");
  });
});

describe("errorsUnder", () => {
  it("matches whole path segments, ignoring the body prefix", async () => {
    const { errorsUnder } = await import("../lib/steps");
    const errors = [
      { field: ["body", "data", "base_dir"] },
      { field: ["training", "epochs"] },
      { field: ["training_extra"] },
    ];
    expect(errorsUnder(errors, ["data"])).toBe(1);
    expect(errorsUnder(errors, ["training"])).toBe(1);
    expect(errorsUnder(errors, ["model"])).toBe(0);
  });
});
