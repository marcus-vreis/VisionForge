import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { pt } from "../i18n/pt";
import { LanguageContext } from "../i18n/context";
import { BottomBar } from "./BottomBar";
import { JobRow } from "./QueueOverlay";

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };
const job = {
  run_id: "r1",
  label: "run",
  task: "detection",
  strategy: "simple",
  submitted_at: "2026-11-02T00:00:00Z",
};

function row(running: boolean) {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={lang}>
      <JobRow job={job} running={running} onCancel={() => {}} onWatch={() => {}} />
    </LanguageContext.Provider>,
  );
}

function bar(queueMode: boolean, lockReason: string | null = null) {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={lang}>
      <BottomBar
        onHistory={() => {}}
        onDatasets={() => {}}
        onQueue={() => {}}
        onTrain={() => {}}
        disabled={false}
        historyCount={0}
        queuedCount={0}
        serverRunning={queueMode}
        selection={{ kind: "cpu", gpu_ids: null }}
        onSelectionChange={() => {}}
        isRunning={queueMode}
        queueMode={queueMode}
        lockReason={lockReason}
        notice={queueMode ? pt.bottomBar.enqueued : null}
      />
    </LanguageContext.Provider>,
  );
}

describe("queue from the same tab (ADR-159)", () => {
  it("offers the training sheet only on the running job", () => {
    expect(row(true)).toContain(pt.queueOverlay.watch);
    expect(row(false)).not.toContain(pt.queueOverlay.watch);
  });

  it("turns the main button into Colocar na fila while the server is busy", () => {
    const busy = bar(true);
    expect(busy).toContain(pt.bottomBar.enqueue);
    expect(busy).toContain(pt.bottomBar.enqueued);
    expect(bar(false)).not.toContain(pt.bottomBar.enqueue);
  });

  it("keeps the Review lock on a queued submission", () => {
    const train = bar(true, pt.bottomBar.trainLockedReview).match(
      /<button[^>]*data-tour="train"[^>]*>/,
    )?.[0];
    expect(train).toContain("disabled");
  });
});
