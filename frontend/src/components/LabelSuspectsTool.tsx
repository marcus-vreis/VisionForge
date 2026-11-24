import { useEffect, useState } from "react";
import { labelSuspectsRun, labelSuspectsSaved } from "../api/client";
import type { LabelSuspectsReport, SuspectSplit, SuspectSplitReport } from "../lib/label-suspects";
import { useT } from "../i18n/useT";
import { Toggle } from "./controls";
import { ActionButton } from "./ui/ActionButton";
import { DataTable, LearnMore } from "./DatasetLabParts";
import { danger, mono, sectionLabel, text, warn } from "./datasetExplorerStyles";

const ORDER: SuspectSplit[] = ["val", "test", "train"];
/** Suspects shown per split; the full list is in the run folder's CSV. */
const SHOWN = 30;

function pct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

function SplitView({ split, report }: { split: SuspectSplit; report: SuspectSplitReport }) {
  const t = useT().labelSuspects;
  return (
    <div style={{ display: "grid", gap: 8 }} data-testid={`suspects-${split}`}>
      <div style={sectionLabel}>{t.splitNames[split]}</div>
      <div style={report.suspect_count > 0 ? warn : text}>
        {t.found(report.suspect_count, report.images)}
        {report.failed > 0 && ` ${t.failed(report.failed)}`}
      </div>
      {report.in_sample && <div style={text}>{t.inSample}</div>}
      {report.suspects.length > 0 && (
        <DataTable>
          <thead>
            <tr>
              <th style={{ textAlign: "left" }}>{t.image}</th>
              <th style={{ textAlign: "left" }}>{t.given}</th>
              <th style={{ textAlign: "left" }}>{t.suspected}</th>
              <th style={{ textAlign: "right" }}>{t.pGiven}</th>
              <th style={{ textAlign: "right" }}>{t.pSuspected}</th>
            </tr>
          </thead>
          <tbody>
            {report.suspects.slice(0, SHOWN).map((s) => (
              <tr key={s.path}>
                <td style={{ wordBreak: "break-all" }}>{s.path}</td>
                <td>{s.given}</td>
                <td>{s.suspected}</td>
                <td style={{ textAlign: "right" }}>{pct(s.p_given)}</td>
                <td style={{ textAlign: "right" }}>{pct(s.p_suspected)}</td>
              </tr>
            ))}
          </tbody>
        </DataTable>
      )}
      {report.suspect_count > SHOWN && <div style={mono}>{t.more(report.suspect_count - SHOWN, split)}</div>}
      {report.pairs.length > 0 && (
        <LearnMore summary={t.pairsTitle}>
          <div style={mono}>
            {report.pairs.map((p) => (
              <div key={`${p.given}>${p.taken_for}`}>{t.pair(p.given, p.taken_for, p.images)}</div>
            ))}
          </div>
        </LearnMore>
      )}
    </div>
  );
}

/** Report of the images whose label the run's model disputes (ADR-237). */
export function LabelSuspectsReportView({ report }: { report: LabelSuspectsReport }) {
  const t = useT().labelSuspects;
  const splits = ORDER.filter((s) => report.splits[s]);
  return (
    <div style={{ display: "grid", gap: 14 }} data-testid="suspects-report">
      <div style={text}>{t.computedAt(new Date(report.computed_at).toLocaleString())}</div>
      {report.missing_splits.length > 0 && (
        <div style={warn}>{t.missing(report.missing_splits.map((s) => t.splitNames[s]).join(", "))}</div>
      )}
      {splits.map((s) => (
        <SplitView key={s} split={s} report={report.splits[s]!} />
      ))}
    </div>
  );
}

/** The form: which splits, the run button, and the last report. */
export function LabelSuspectsTool({ runId }: { runId: string }) {
  const t = useT().labelSuspects;
  const [chosen, setChosen] = useState<Record<SuspectSplit, boolean>>({ val: true, test: true, train: false });
  const [report, setReport] = useState<{ runId: string; report: LabelSuspectsReport | null } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    labelSuspectsSaved(runId)
      .then((r) => {
        if (live) setReport({ runId, report: r });
      })
      // No kept report is not an error worth a message.
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [runId]);

  const splits = ORDER.filter((s) => chosen[s]);
  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      setReport({ runId, report: await labelSuspectsRun(runId, splits) });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const shown = report?.runId === runId ? report.report : null;
  return (
    <div style={{ display: "grid", gap: 12 }} data-testid="label-suspects">
      <div style={text}>{t.help}</div>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        {ORDER.map((s) => (
          <Toggle
            key={s}
            label={t.splitNames[s]}
            value={chosen[s]}
            onChange={(v) => setChosen((c) => ({ ...c, [s]: v }))}
            help={s === "train" ? t.inSample : undefined}
          />
        ))}
      </div>
      <div>
        <ActionButton onClick={() => void run()} disabled={busy || splits.length === 0} aria-busy={busy}>
          {busy ? t.running : t.start}
        </ActionButton>
      </div>
      {busy && <div style={text}>{t.runningHint}</div>}
      {error && <div style={danger}>{t.error(error)}</div>}
      {shown && <LabelSuspectsReportView report={shown} />}
    </div>
  );
}
