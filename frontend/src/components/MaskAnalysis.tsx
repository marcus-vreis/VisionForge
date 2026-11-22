import { explorerScanCancel, maskAnalysisSaved, maskAnalysisStart, maskAnalysisStatus } from "../api/client";
import { useAnalysisJob } from "../lib/analysis-job";
import type { MaskReport, MaskRequest } from "../lib/mask-analysis";
import { coverageLabel, sharePct as pct } from "../lib/mask-analysis";
import { useT } from "../i18n/useT";
import { ROUTE, useRouteOpen } from "../lib/session";
import { AnalysisJobCard } from "./AnalysisJobCard";
import { BarList, DataTable, LearnMore } from "./DatasetLabParts";
import { card, mono, sectionLabel, text, warn } from "./datasetExplorerStyles";

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

const MASK_API = {
  start: maskAnalysisStart,
  status: maskAnalysisStatus,
  saved: maskAnalysisSaved,
  cancel: explorerScanCancel,
};

function MaskAnalysisInner({ request }: { request: MaskRequest }) {
  const t = useT().maskAnalysis;
  const ready = request.base_dir.trim() !== "";
  const state = useAnalysisJob(MASK_API, request, ready);
  return (
    <section data-testid="mask-analysis" style={{ display: "grid", gap: 12, marginBottom: 18 }}>
      <AnalysisJobCard words={t} state={state} ready={ready} />
      {state.shown && <MaskReportView report={state.shown} />}
    </section>
  );
}

/** Absent where the server does not run it (a shared server, ADR-216). */
export function MaskAnalysis({ request }: { request: MaskRequest }) {
  return useRouteOpen(ROUTE.maskAnalysis) ? <MaskAnalysisInner request={request} /> : null;
}
