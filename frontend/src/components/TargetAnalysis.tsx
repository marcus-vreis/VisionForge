import { explorerScanCancel, targetAnalysisSaved, targetAnalysisStart, targetAnalysisStatus } from "../api/client";
import { useAnalysisJob } from "../lib/analysis-job";
import { num, SHIFT_WARN_SD, type TargetReport, type TargetReportItem, type TargetRequest } from "../lib/target-analysis";
import { useT } from "../i18n/useT";
import { ROUTE, useRouteOpen } from "../lib/session";
import { AnalysisJobCard } from "./AnalysisJobCard";
import { ImageQualityCard } from "./ImageQualityCard";
import { DataTable, LearnMore } from "./DatasetLabParts";
import { card, mono, sectionLabel, stepTitle, text, warn } from "./datasetExplorerStyles";

const SPLITS = ["train", "val", "test"] as const;
// One stroke per split; dashes keep them apart without relying on color alone.
const STROKE: Record<string, { color: string; dash?: string }> = {
  train: { color: "var(--accent-vf)" },
  val: { color: "var(--vf-text)", dash: "5 3" },
  test: { color: "var(--vf-text-dim)", dash: "1 3" },
};
const W = 320;
const H = 90;

/** Each split's histogram as a share of its own values, on the target's shared bins. */
function Histogram({ item, label }: { item: TargetReportItem; label: string }) {
  const lines = SPLITS.flatMap((s) => {
    const sp = item.splits[s];
    const total = sp ? sp.histogram.reduce((a, b) => a + b, 0) : 0;
    return sp && total > 0 ? [{ s, shares: sp.histogram.map((n) => n / total) }] : [];
  });
  const top = Math.max(0, ...lines.flatMap((l) => l.shares));
  const x = (k: number) => ((k + 0.5) / item.bins) * W;
  const y = (v: number) => H - (top > 0 ? (v / top) * (H - 6) : 0);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} style={{ width: "100%", maxWidth: 420, height: "auto" }}>
      <line x1={0} y1={H - 0.5} x2={W} y2={H - 0.5} stroke="var(--vf-panel-stroke)" />
      {lines.map((l) => (
        <polyline
          key={l.s}
          data-split={l.s}
          fill="none"
          stroke={STROKE[l.s].color}
          strokeDasharray={STROKE[l.s].dash}
          strokeWidth={1.6}
          points={l.shares.map((v, k) => `${x(k).toFixed(1)},${y(v).toFixed(1)}`).join(" ")}
        />
      ))}
    </svg>
  );
}

function TargetBlock({ item, splitNames }: { item: TargetReportItem; splitNames: Record<string, string> }) {
  const t = useT().targetAnalysis;
  const splits = SPLITS.filter((s) => item.splits[s]);
  const shifted = splits.filter((s) => {
    const shift = item.splits[s]?.shift_sd;
    return s !== "train" && shift !== null && shift !== undefined && Math.abs(shift) >= SHIFT_WARN_SD;
  });
  const outliers = splits.flatMap((s) => (item.splits[s]?.outliers ?? []).map((o) => ({ ...o, split: s })));
  return (
    <div style={{ display: "grid", gap: 8 }}>
      <div style={sectionLabel}>{item.name}</div>
      <DataTable>
        <thead>
          <tr>
            <th style={{ textAlign: "left" }}>{t.splitColumn}</th>
            {[t.n, t.bad, t.mean, t.std, t.min, t.median, t.max, t.outliers, t.shift].map((h) => (
              <th key={h} style={{ textAlign: "right" }}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {splits.map((s) => {
            const sp = item.splits[s]!;
            return (
              <tr key={s}>
                <td>{splitNames[s]}</td>
                {[sp.count, sp.bad].map((v, i) => (
                  <td key={i} style={{ textAlign: "right" }}>
                    {v}
                  </td>
                ))}
                {[sp.mean, sp.std, sp.min, sp.median, sp.max].map((v, i) => (
                  <td key={`s${i}`} style={{ textAlign: "right" }}>
                    {num(v)}
                  </td>
                ))}
                <td style={{ textAlign: "right" }}>{sp.outlier_count}</td>
                <td style={{ textAlign: "right" }}>{s === "train" ? "—" : num(sp.shift_sd)}</td>
              </tr>
            );
          })}
        </tbody>
      </DataTable>
      <Histogram item={item} label={t.histogramLabel(item.name)} />
      <div style={mono}>
        {t.range(num(item.range[0]), num(item.range[1]))}
        {"  ·  "}
        {splits.map((s) => `${splitNames[s]}: ${t.strokes[s]}`).join("  ·  ")}
      </div>
      {shifted.map((s) => (
        <div key={s} style={warn}>
          {t.shifted(splitNames[s], num(item.splits[s]?.shift_sd))}
        </div>
      ))}
      {item.fences && outliers.length > 0 && (
        <LearnMore summary={t.outliersList(outliers.length, num(item.fences[0]), num(item.fences[1]))}>
          <div style={mono}>
            {outliers.slice(0, 20).map((o) => (
              <div key={`${o.split}:${o.line}`} style={{ wordBreak: "break-all" }}>
                [{splitNames[o.split]}] {t.line(o.line)} · {o.image || "—"} · {num(o.value)}
              </div>
            ))}
          </div>
        </LearnMore>
      )}
    </div>
  );
}

/** What the targets look like (ADR-235). */
export function TargetReportView({ report }: { report: TargetReport }) {
  const t = useT().targetAnalysis;
  const splitNames: Record<string, string> = useT().analysisJob.splitNames;
  const present = SPLITS.filter((s) => report.splits[s]?.present);
  const warnings = present.flatMap((s) => {
    const sp = report.splits[s];
    const lines: string[] = [];
    if (sp.missing_columns.length) lines.push(t.missingColumns(splitNames[s], sp.missing_columns.join(", ")));
    if (sp.missing_images) lines.push(t.missingImages(splitNames[s], sp.missing_images, sp.rows));
    return lines;
  });
  const badCells = present.flatMap((s) => report.splits[s].bad_examples.map((e) => ({ ...e, split: s })));
  const missing = present.flatMap((s) => report.splits[s].missing_examples.map((e) => ({ ...e, split: s })));
  return (
    <div style={card} data-testid="target-report">
      <div style={report.cancelled ? warn : text} role="status">
        {report.cancelled ? t.cancelled : t.complete(present.map((s) => `${splitNames[s]} ${report.splits[s].rows}`).join(", "))}
      </div>
      {warnings.length === 0 && badCells.length === 0 ? (
        <div style={text}>{t.noProblems}</div>
      ) : (
        warnings.map((w) => (
          <div key={w} style={warn}>
            {w}
          </div>
        ))
      )}
      {badCells.length > 0 && <div style={warn}>{t.badCells(badCells.length)}</div>}
      {report.targets.map((item) => (
        <TargetBlock key={item.name} item={item} splitNames={splitNames} />
      ))}
      <div style={text}>{t.tukeyHelp(report.tukey_k)}</div>
      {report.correlations.length > 0 && (
        <>
          <h4 style={{ ...stepTitle, margin: 0, fontSize: 13 }}>{t.correlationTitle}</h4>
          <div style={mono}>
            {report.correlations.map((c) => (
              <div key={`${c.a}+${c.b}`}>{t.correlation(c.a, c.b, c.r === null ? "—" : c.r.toFixed(3))}</div>
            ))}
          </div>
          <div style={text}>{t.correlationHelp}</div>
        </>
      )}
      {(badCells.length > 0 || missing.length > 0) && (
        <LearnMore summary={t.rowsTitle}>
          <div style={mono}>
            {badCells.slice(0, 20).map((e) => (
              <div key={`b${e.split}:${e.line}:${e.target}`}>
                [{splitNames[e.split]}] {t.line(e.line)} · {e.target} = “{e.cell}”
              </div>
            ))}
            {missing.slice(0, 20).map((e) => (
              <div key={`m${e.split}:${e.line}`} style={{ wordBreak: "break-all" }}>
                [{splitNames[e.split]}] {t.line(e.line)} · {t.noImage(e.image)}
              </div>
            ))}
          </div>
        </LearnMore>
      )}
    </div>
  );
}

const TARGET_API = {
  start: targetAnalysisStart,
  status: targetAnalysisStatus,
  saved: targetAnalysisSaved,
  cancel: explorerScanCancel,
};

function TargetAnalysisInner({ request }: { request: TargetRequest }) {
  const t = useT().targetAnalysis;
  const ready = request.base_dir.trim() !== "" && request.target_columns.length > 0;
  const state = useAnalysisJob(TARGET_API, request, ready);
  return (
    <section data-testid="target-analysis" style={{ display: "grid", gap: 12, marginTop: 18 }}>
      <AnalysisJobCard words={t} state={state} ready={ready} />
      {state.shown && <TargetReportView report={state.shown} />}
      {state.shown?.quality && state.shown.quality.count > 0 && (
        <ImageQualityCard quality={state.shown.quality} root={request.base_dir} />
      )}
    </section>
  );
}

/** Absent where the server does not run it (a shared server, ADR-216). */
export function TargetAnalysis({ request }: { request: TargetRequest }) {
  return useRouteOpen(ROUTE.targetAnalysis) ? <TargetAnalysisInner request={request} /> : null;
}
