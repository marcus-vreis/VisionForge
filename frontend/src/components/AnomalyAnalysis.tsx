import { anomalyAnalysisSaved, anomalyAnalysisStart, anomalyAnalysisStatus, explorerScanCancel } from "../api/client";
import { DEFECT_PREFIX, type AnomalyReport, type AnomalyRequest, type LookAlike } from "../lib/anomaly-analysis";
import { useAnalysisJob } from "../lib/analysis-job";
import { useT } from "../i18n/useT";
import { ROUTE, useRouteOpen } from "../lib/session";
import { AnalysisJobCard } from "./AnalysisJobCard";
import { ImageQualityCard } from "./ImageQualityCard";
import { DataTable, LearnMore } from "./DatasetLabParts";
import { card, mono, sectionLabel, text, warn } from "./datasetExplorerStyles";

function LookAlikes({ items, title }: { items: LookAlike[]; title: string }) {
  if (items.length === 0) return null;
  return (
    <LearnMore summary={title}>
      <div style={mono}>
        {items.slice(0, 20).map((m) => (
          <div key={`${m.path}|${m.like}`} style={{ wordBreak: "break-all" }}>
            {m.path} ≈ {m.like} ({m.distance})
          </div>
        ))}
      </div>
    </LearnMore>
  );
}

/** Train, normal test or the defect kind, as the screen names a group. */
function useGroupName(): (group: string) => string {
  const t = useT().anomalyAnalysis;
  return (group: string) =>
    group === "train" ? t.train : group === "test_normal" ? t.testNormal : t.defect(group.slice(DEFECT_PREFIX.length));
}

/** What the anomaly dataset holds (ADR-236). */
export function AnomalyReportView({ report }: { report: AnomalyReport }) {
  const t = useT().anomalyAnalysis;
  const name = useGroupName();
  const defects = Object.entries(report.counts).filter(([g]) => g.startsWith(DEFECT_PREFIX));
  const anomalous = defects.reduce((sum, [, n]) => sum + n, 0);
  const few = defects.filter(([, n]) => n < report.few_images).map(([g, n]) => `${g.slice(DEFECT_PREFIX.length)} (${n})`);
  const warnings = [
    !report.train_present && t.noTrain,
    defects.length === 0 && t.noDefects,
    report.extra_train_folders.length > 0 && t.extraTrain(report.extra_train_folders.join(", ")),
    few.length > 0 && t.few(few.join(", "), report.few_images),
    report.train_like_defect_count > 0 && t.trainLikeDefect(report.train_like_defect_count),
    report.test_like_train_count > 0 && t.testLikeTrain(report.test_like_train_count),
    report.size_mismatch.length > 0 && t.sizeMismatch(report.size_mismatch.map(name).join(", ")),
    report.mode_mismatch.length > 0 && t.modeMismatch(report.mode_mismatch.map(name).join(", ")),
    report.missing_masks && report.missing_masks.count > 0 && t.missingMasks(report.missing_masks.count),
    report.unreadable_count > 0 && t.unreadable(report.unreadable_count),
  ].filter((w): w is string => typeof w === "string");
  return (
    <div style={card} data-testid="anomaly-report">
      <div style={report.sampled || report.cancelled ? warn : text} role="status">
        {report.cancelled
          ? t.cancelled(report.scanned)
          : report.sampled
            ? t.sampled(report.scanned, report.total_images)
            : t.complete(report.scanned)}
      </div>
      <div style={mono}>
        {t.totals(report.counts.train ?? 0, report.counts.test_normal ?? 0, anomalous, defects.length)}
      </div>
      {warnings.length === 0 ? (
        <div style={text}>{t.noProblems}</div>
      ) : (
        warnings.map((w) => (
          <div key={w} style={warn}>
            {w}
          </div>
        ))
      )}
      <div style={sectionLabel}>{t.groupsTitle}</div>
      <DataTable>
        <thead>
          <tr>
            <th style={{ textAlign: "left" }}>{t.groupColumn}</th>
            <th style={{ textAlign: "right" }}>{t.imagesColumn}</th>
            <th style={{ textAlign: "left" }}>{t.sizeColumn}</th>
            <th style={{ textAlign: "left" }}>{t.modeColumn}</th>
          </tr>
        </thead>
        <tbody>
          {report.groups.map((g) => (
            <tr key={g.group}>
              <td>{name(g.group)}</td>
              <td style={{ textAlign: "right" }}>{report.counts[g.group] ?? 0}</td>
              <td>
                {g.sizes.map((s) => `${s.size} (${s.count})`).join(", ")}
                {g.distinct_sizes > g.sizes.length ? ` +${g.distinct_sizes - g.sizes.length}` : ""}
              </td>
              <td>
                {Object.entries(g.modes)
                  .map(([m, n]) => `${m} (${n})`)
                  .join(", ")}
              </td>
            </tr>
          ))}
        </tbody>
      </DataTable>
      <div style={text}>{t.lookHelp(report.max_distance)}</div>
      <LookAlikes items={report.train_like_defect} title={t.trainLikeDefectList} />
      <LookAlikes items={report.test_like_train} title={t.testLikeTrainList} />
      {report.missing_masks && report.missing_masks.paths.length > 0 && (
        <LearnMore summary={t.missingMasksList}>
          <div style={mono}>
            {report.missing_masks.paths.slice(0, 20).map((p) => (
              <div key={p} style={{ wordBreak: "break-all" }}>
                {p}
              </div>
            ))}
          </div>
        </LearnMore>
      )}
    </div>
  );
}

const ANOMALY_API = {
  start: anomalyAnalysisStart,
  status: anomalyAnalysisStatus,
  saved: anomalyAnalysisSaved,
  cancel: explorerScanCancel,
};

function AnomalyAnalysisInner({ request }: { request: AnomalyRequest }) {
  const t = useT().anomalyAnalysis;
  const ready = request.base_dir.trim() !== "";
  const state = useAnalysisJob(ANOMALY_API, request, ready);
  const name = useGroupName();
  return (
    <section data-testid="anomaly-analysis" style={{ display: "grid", gap: 12, marginTop: 18 }}>
      <AnalysisJobCard words={t} state={state} ready={ready} />
      {state.shown && <AnomalyReportView report={state.shown} />}
      {state.shown?.quality && state.shown.quality.count > 0 && (
        <ImageQualityCard quality={state.shown.quality} root={request.base_dir} splitLabel={name} />
      )}
    </section>
  );
}

/** Absent where the server does not run it (a shared server, ADR-216). */
export function AnomalyAnalysis({ request }: { request: AnomalyRequest }) {
  return useRouteOpen(ROUTE.anomalyAnalysis) ? <AnomalyAnalysisInner request={request} /> : null;
}
