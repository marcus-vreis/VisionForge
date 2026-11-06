import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { LanguageContext } from "../i18n/context";
import { BottomBar } from "./BottomBar";
import { JobRow, QueueLaneList } from "./QueueOverlay";

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

describe("one section per queue (ADR-163 phase 4)", () => {
  const COLAB = "colab:ana@gmail.com";
  const local = { ...job, run_id: "local-run", label: "local_run", lane: "local" };
  const colab = {
    ...job,
    run_id: "colab-run",
    label: "colab_run",
    task: "classification",
    lane: COLAB,
    stop_at: "vm" as const,
  };
  const waiting = { ...colab, run_id: "colab-next", label: "colab_next", stop_at: undefined };

  function lanes(showLanes: boolean) {
    return renderToStaticMarkup(
      <LanguageContext.Provider value={lang}>
        <QueueLaneList
          lanes={[
            { lane: "local", active: local, pending: [] },
            { lane: COLAB, active: colab, pending: [waiting] },
          ]}
          showLanes={showLanes}
          showProfile={false}
          busyId={null}
          onCancel={() => {}}
          onWatch={() => {}}
        />
      </LanguageContext.Provider>,
    );
  }

  function section(html: string, lane: string): string {
    return html.match(new RegExp(`<section[^>]*data-lane="${lane}"[^>]*>.*?</section>`))?.[0] ?? "";
  }

  it("heads each queue with where it runs, the Colab one with its account", () => {
    const html = lanes(true);
    expect(html).toContain(`<span>${pt.queueOverlay.laneLocal}</span>`);
    expect(html).toContain(`<span>${pt.queueOverlay.laneColab("ana@gmail.com")}</span>`);
    // A lone local queue has no visible heading, only its accessible name.
    const lone = lanes(false);
    expect(lone).not.toContain(`<span>${pt.queueOverlay.laneLocal}</span>`);
    expect(lone).toContain(`aria-label="${pt.queueOverlay.laneLocal}"`);
  });

  it("keeps each job in its own queue's section, with its origin on the row", () => {
    const html = lanes(true);
    const l = section(html, "local");
    const c = section(html, COLAB);
    expect(l).toContain("local_run");
    expect(l).not.toContain("colab_run");
    expect(c).toContain("colab_run");
    expect(c).toContain("colab_next");
    expect(c).not.toContain("local_run");
    expect(l).toContain(`${pt.queueOverlay.originLocal} · `);
    expect(c).toContain(`${pt.queueOverlay.originColab} · `);
  });

  it("gives both running jobs their own stop and watch, and the waiting one its place", () => {
    const html = lanes(true);
    for (const lane of ["local", COLAB]) {
      const s = section(html, lane);
      expect(s).toContain(pt.queueOverlay.stop);
      expect(s).toContain(pt.queueOverlay.watch);
    }
    expect(section(html, COLAB)).toContain(pt.queueOverlay.position(1));
  });

  it("says an empty queue is empty", () => {
    const html = renderToStaticMarkup(
      <LanguageContext.Provider value={lang}>
        <QueueLaneList
          lanes={[
            { lane: "local", active: local, pending: [] },
            { lane: COLAB, active: null, pending: [] },
          ]}
          showLanes
          showProfile={false}
          busyId={null}
          onCancel={() => {}}
        />
      </LanguageContext.Provider>,
    );
    expect(section(html, COLAB)).toContain(pt.queueOverlay.laneEmpty);
  });
});

describe("a Colab job waiting (ADR-163 phase 4)", () => {
  it("waits for its account, not for this machine's GPU, in both languages", () => {
    const ptNote = pt.trainingOverlay.queuedColabNote("ana@gmail.com", 2);
    expect(ptNote).toContain("ana@gmail.com");
    expect(ptNote).toContain("2º na fila");
    expect(ptNote).not.toContain("GPU");
    const enNote = en.trainingOverlay.queuedColabNote("ana@gmail.com", undefined);
    expect(enNote).toContain("ana@gmail.com");
    expect(enNote).not.toContain("GPU");
  });
});
