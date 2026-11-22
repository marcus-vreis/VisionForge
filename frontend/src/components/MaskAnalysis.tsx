import { useCallback, useEffect, useState } from "react";
import { explorerScanCancel, maskAnalysisSaved, maskAnalysisStart, maskAnalysisStatus } from "../api/client";
import { isScanFinished, scanPercent } from "../lib/dataset-explorer";
import type { MaskAnalysisJob, MaskReport, MaskRequest, SavedMaskAnalysis } from "../lib/mask-analysis";
import { coverageLabel, sharePct as pct } from "../lib/mask-analysis";
import { useT } from "../i18n/useT";
import { ROUTE, useRouteOpen } from "../lib/session";
import { ActionButton } from "./ui/ActionButton";
import { BarList, DataTable, LearnMore } from "./DatasetLabParts";
import { card, danger, ghostButton, mono, sectionLabel, stepTitle, text, warn } from "./datasetExplorerStyles";

const SPLITS = ["train", "val", "test"] as const;

/** What the masks hold (ADR-234). */
export function MaskReportView({ report }: { report: MaskReport }) {
  const t = useT().maskAnalysis;
  const splits = SPLITS.filter((s) => report.splits[s]?.present);
  const unpaired = splits.flatMap((s) => {
    const info = report.splits[s];
    return info.no_mask_count + info.no_image_count > 0
      ? [t.unpaired(t.splitNames[s], info.no_mask_count, info.no_image_count)]
      : [];
  });
  const maxCoverage = report.coverage_bins.reduce((m, b) => Math.max(m, b.count), 0);
  const problems = [
    report.out_of_range_pixels > 0 && t.outOfRange(report.out_of_range_ids.join(", "), report.out_of_range_pixels, report.num_classes - 1),
    report.color_masks > 0 && t.color(report.color_masks),
    report.size_mismatch > 0 && t.sizeMismatch(report.size_mismatch),
    report.unreadable > 0 && t.unreadable(report.unreadable),
    report.absent_classes.length > 0 && t.absent(report.absent_classes.join(", ")),
  ].filter((p): p is string => typeof p === "string");
  return (
    <div style={card} data-testid="mask-report">
      <div style={report.sampled || report.cancelled ? warn : text} role="status">
        {report.cancelled
          ? t.cancelled(report.scanned)
          : report.sampled
            ? t.sampled(report.scanned, report.total_pairs)
            : t.complete(report.scanned)}
      </div>
      {unpaired.map((line) => (
        <div key={line} style={warn}>
          {line}
        </div>
      ))}
      {problems.length === 0 ? (
        <div style={text}>{t.noProblems}</div>
      ) : (
        problems.map((line) => (
          <div key={line} style={warn}>
            {line}
          </div>
        ))
      )}
      <div style={sectionLabel}>{t.classesTitle}</div>
      <DataTable>
        <thead>
          <tr>
            <th style={{ textAlign: "left" }}>{t.classColumn}</th>
            <th style={{ textAlign: "right" }}>{t.allColumn}</th>
            {splits.map((s) => (
              <th key={s} style={{ textAlign: "right" }}>
                {t.splitNames[s]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {report.classes.map((c) => (
            <tr key={c.id}>
              <td>{c.ignore ? t.ignoreId(c.id) : c.out_of_range ? t.badId(c.id) : String(c.id)}</td>
              <td style={{ textAlign: "right" }}>{pct(c.share)}</td>
              {splits.map((s) => (
                <td key={s} style={{ textAlign: "right" }}>
                  {c.shares[s] === undefined ? "—" : pct(c.shares[s])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </DataTable>
      <div style={text}>{t.classesHelp}</div>
      <div style={sectionLabel}>{t.coverageTitle}</div>
      <BarList
        bars={report.coverage_bins.map((b) => ({
          label: coverageLabel(b, t.coverageEmpty),
          value: b.count,
          pct: maxCoverage > 0 ? (b.count / maxCoverage) * 100 : 0,
        }))}
      />
      <div style={text}>{t.coverageHelp}</div>
      <div style={sectionLabel}>{t.tinyTitle}</div>
      <div style={report.tiny_images > 0 ? warn : text}>
        {report.tiny_images > 0
          ? t.tiny(
              report.tiny_images,
              report.tiny_regions.map((r) => t.tinyClass(r.id, r.regions)).join(", "),
              report.tiny_share * 100,
            )
          : t.noTiny(report.tiny_share * 100)}
      </div>
      {Object.keys(report.examples).length > 0 && (
        <LearnMore summary={t.examplesTitle}>
          <div style={mono}>
            {Object.entries(report.examples).flatMap(([kind, items]) =>
              items.slice(0, 10).map((e) => (
                <div key={`${kind}:${e.path}`} style={{ wordBreak: "break-all" }}>
                  [{t.splitNames[e.split as keyof typeof t.splitNames] ?? e.split}] {e.path} ·{" "}
                  {t.kinds[kind as keyof typeof t.kinds] ?? kind}
                </div>
              )),
            )}
          </div>
        </LearnMore>
      )}
    </div>
  );
}

function MaskAnalysisInner({ request }: { request: MaskRequest }) {
  const t = useT().maskAnalysis;
  const [job, setJob] = useState<MaskAnalysisJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [kept, setKept] = useState<{ key: string; saved: SavedMaskAnalysis } | null>(null);
  const key = JSON.stringify(request);
  const saved = kept?.key === key ? kept.saved : null;
  const ready = request.base_dir.trim() !== "";
  const jobId = job?.job_id ?? null;
  const running = job !== null && !isScanFinished(job.state);

  useEffect(() => {
    if (!ready) return;
    let live = true;
    maskAnalysisSaved(request)
      .then((s) => {
        if (live) setKept({ key, saved: s });
      })
      // No kept analysis is not an error worth a message.
      .catch(() => undefined);
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the request
  }, [key, ready]);

  useEffect(() => {
    if (!jobId || !running) return;
    const timer = setInterval(() => {
      maskAnalysisStatus(jobId)
        .then((next) => {
          setJob(next);
          if (isScanFinished(next.state)) setCancelling(false);
        })
        .catch((e: unknown) => {
          setError(e instanceof Error ? e.message : String(e));
          setCancelling(false);
        });
    }, 600);
    return () => clearInterval(timer);
  }, [jobId, running]);

  const start = useCallback(async () => {
    setError(null);
    setJob(null);
    try {
      setJob(await maskAnalysisStart(request));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [request]);

  const cancel = async () => {
    if (!jobId) return;
    setCancelling(true);
    try {
      await explorerScanCancel(jobId);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setCancelling(false);
    }
  };

  const done = job ? scanPercent(job.done, job.total) : 0;
  const shown = job?.result ?? (!running ? saved?.report : null);
  return (
    <section data-testid="mask-analysis" style={{ display: "grid", gap: 12, marginBottom: 18 }}>
      <div style={card}>
        <h3 style={{ ...stepTitle, margin: 0 }}>{t.title}</h3>
        <div style={text}>{t.help}</div>
        <div>
          {running ? (
            <button type="button" style={ghostButton} onClick={() => void cancel()} disabled={cancelling}>
              {cancelling ? t.cancelling : t.cancel}
            </button>
          ) : (
            <ActionButton onClick={() => void start()} disabled={!ready}>
              {t.start}
            </ActionButton>
          )}
        </div>
        {running && job && (
          <div>
            <div style={mono}>{t.progress(job.done, job.total)}</div>
            <div
              role="progressbar"
              aria-valuenow={done}
              aria-valuemin={0}
              aria-valuemax={100}
              style={{ height: 6, borderRadius: 3, background: "rgb(var(--lt-ink, 255 255 255) / 0.08)" }}
            >
              <div style={{ height: "100%", width: `${done}%`, borderRadius: 3, background: "var(--accent-vf)" }} />
            </div>
          </div>
        )}
        {error && <div style={danger}>{t.failed(error)}</div>}
        {job?.state === "failed" && job.error && <div style={danger}>{t.failed(job.error)}</div>}
        {!job && saved?.report && (
          <div style={saved.stale ? warn : text} role="status">
            {t.savedAt(saved.saved_at ? new Date(saved.saved_at).toLocaleString() : "")}
            {saved.stale && ` ${t.savedStale}`}
          </div>
        )}
      </div>
      {shown && <MaskReportView report={shown} />}
    </section>
  );
}

/** Absent where the server does not run it (a shared server, ADR-216). */
export function MaskAnalysis({ request }: { request: MaskRequest }) {
  return useRouteOpen(ROUTE.maskAnalysis) ? <MaskAnalysisInner request={request} /> : null;
}
