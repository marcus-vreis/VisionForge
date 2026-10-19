import type { CSSProperties, ReactNode } from "react";
import { stdDdof } from "../lib/cv-std";
import {
  bestByMean,
  childTarget,
  fewestSeeds,
  formatAggregate,
  formatNumber,
  formatOverrides,
  formatP,
  hasFewSeeds,
  holmVerdict,
} from "../lib/run-groups";
import { STOPPED_COLOR, unitState } from "../lib/unit-status";
import { useT } from "../i18n/useT";
import type { GroupChild, GroupVariant, MetricAggregate, PairedTest, RunGroup } from "../types/run";

interface RunGroupDetailProps {
  group: RunGroup;
  /** The task's color, as the history card wears it (lib/task-accent). */
  accent: string;
  /** Run ids History lists: a seed deleted since the group was written is
   *  named but not linked. */
  knownRunIds?: ReadonlySet<string>;
  onOpenRun?: (runId: string) => void;
}

const OK_COLOR = "oklch(calc(0.85 - var(--lt-flip, 0) * 0.4) 0.16 150)";
const FAILED_COLOR = "oklch(calc(0.85 - var(--lt-flip, 0) * 0.4) 0.14 22)";

const thStyle: CSSProperties = {
  textAlign: "left",
  padding: "6px 8px",
  borderBottom: "1px solid var(--vf-panel-stroke)",
  fontSize: 9,
  letterSpacing: "0.14em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
  fontWeight: 500,
};

const tdStyle: CSSProperties = {
  padding: "6px 8px",
  borderBottom: "1px solid rgb(var(--lt-ink, 255 255 255) / 0.04)",
  color: "var(--vf-text)",
};

const tdLabelStyle: CSSProperties = {
  ...tdStyle,
  color: "var(--vf-text-muted)",
  fontSize: 10,
};

const tableBoxStyle: CSSProperties = {
  padding: 10,
  background: "rgb(var(--lt-shade, 0 0 0) / 0.30)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 10,
  overflowX: "auto",
};

const tableStyle: CSSProperties = {
  width: "100%",
  borderCollapse: "collapse",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
};

const noteStyle: CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  lineHeight: 1.5,
  color: "var(--vf-text-muted)",
};

function GroupSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div
      style={{
        padding: "14px 16px",
        background: "rgb(var(--lt-ink, 255 255 255) / 0.02)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 12,
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
        }}
      >
        {title}
      </div>
      {children}
    </div>
  );
}

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div
      style={{
        display: "flex",
        gap: 12,
        padding: "6px 0",
        borderTop: "1px solid var(--vf-panel-stroke)",
      }}
    >
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--vf-text-muted)",
          letterSpacing: "0.10em",
          textTransform: "uppercase",
          minWidth: 130,
        }}
      >
        {label}
      </span>
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          color: "var(--vf-text)",
          wordBreak: "break-all",
        }}
      >
        {children}
      </span>
    </div>
  );
}

/** `mean ± half`, the interval's bounds in the tooltip; a dash without a mean. */
function MeanCi({ agg, accent }: { agg: MetricAggregate | undefined; accent?: string }) {
  const shown = formatAggregate(agg);
  if (!shown) return <>—</>;
  return (
    <span
      title={shown.low !== null && shown.high !== null ? `[${shown.low}, ${shown.high}]` : undefined}
    >
      <span style={{ color: accent, fontWeight: 600 }}>
        {shown.mean}
      </span>
      {shown.half !== null && (
        <span style={{ color: "var(--vf-text-muted)" }}>
          {" ± "}
          {shown.half}
        </span>
      )}
    </span>
  );
}

function OpenRun({
  child,
  accent,
  knownRunIds,
  onOpenRun,
}: {
  child: GroupChild;
  accent: string;
  knownRunIds?: ReadonlySet<string>;
  onOpenRun?: (runId: string) => void;
}) {
  const t = useT();
  if (!child.run_id) return <span style={{ color: "var(--vf-text-muted)" }}>{t.runGroup.detail.noRun}</span>;
  const target = childTarget(child, knownRunIds);
  if (target === null || !onOpenRun) {
    return <span style={{ color: "var(--vf-text-muted)" }}>{t.runGroup.detail.removed}</span>;
  }
  return (
    <button
      type="button"
      onClick={() => onOpenRun(target)}
      title={target}
      style={{
        padding: "3px 9px",
        background: "rgb(var(--lt-ink, 255 255 255) / 0.04)",
        border: `1px solid ${accent}`,
        borderRadius: 6,
        color: "var(--vf-text)",
        fontFamily: "var(--font-mono)",
        fontSize: 10,
        letterSpacing: "0.10em",
        textTransform: "uppercase",
        cursor: "pointer",
      }}
    >
      {t.runGroup.detail.open}
    </button>
  );
}

function ChildStatus({ child }: { child: GroupChild }) {
  const t = useT();
  const state = unitState(child.status);
  if (state === "ok") return <span style={{ color: OK_COLOR }}>✓ {t.runGroup.detail.status.ok}</span>;
  if (state === "stopped") {
    return <span style={{ color: STOPPED_COLOR }}>■ {t.runGroup.detail.status.stopped}</span>;
  }
  return (
    <span style={{ color: FAILED_COLOR }}>
      × {t.runGroup.detail.status.failed}
      {child.error ? ` · ${child.error}` : ""}
    </span>
  );
}

/** The seeds of a group (or a K-fold's folds), each with the run it became. */
function SeedsTable({
  seeds,
  accent,
  metric,
  showVariant,
  byFold = false,
  knownRunIds,
  onOpenRun,
}: {
  seeds: GroupChild[];
  accent: string;
  metric: string | null;
  showVariant: boolean;
  byFold?: boolean;
  knownRunIds?: ReadonlySet<string>;
  onOpenRun?: (runId: string) => void;
}) {
  const t = useT();
  const c = t.runGroup.detail.cols;
  return (
    <div style={tableBoxStyle}>
      <table style={tableStyle}>
        <thead>
          <tr>
            {showVariant && <th style={thStyle}>{c.variant}</th>}
            <th style={thStyle}>{byFold ? c.fold : c.seed}</th>
            <th style={thStyle}>{metric ?? c.metric}</th>
            <th style={thStyle}>{c.status}</th>
            <th style={thStyle}>{c.run}</th>
          </tr>
        </thead>
        <tbody>
          {seeds.map((child) => (
            <tr key={`${child.variant ?? ""}:${byFold ? child.fold : child.seed}`}>
              {showVariant && <td style={tdLabelStyle}>{child.variant}</td>}
              <td style={tdLabelStyle}>{byFold ? child.fold : child.seed}</td>
              <td style={tdStyle}>{formatNumber(metric ? child.metrics[metric] : null)}</td>
              <td style={tdStyle}>
                <ChildStatus child={child} />
              </td>
              <td style={tdStyle}>
                <OpenRun child={child} accent={accent} knownRunIds={knownRunIds} onOpenRun={onOpenRun} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ReplicatesBody({ group, accent, knownRunIds, onOpenRun }: RunGroupDetailProps) {
  const t = useT();
  const c = t.runGroup.detail.cols;
  const aggregates = group.aggregates ?? {};
  const names = Object.keys(aggregates);
  // The metric the job was about first, the others in the order the report had them.
  const ordered = group.metric && names.includes(group.metric)
    ? [group.metric, ...names.filter((n) => n !== group.metric)]
    : names;
  const headline = group.metric ? aggregates[group.metric] : undefined;
  // A K-fold reads like a replicate set, with folds where the seeds are (ADR-123).
  const kfold = group.kind === "kfold";

  return (
    <>
      {ordered.length > 0 && (
        <GroupSection title={t.runGroup.detail.aggregateTitle}>
          <div style={tableBoxStyle}>
            <table style={tableStyle}>
              <thead>
                <tr>
                  <th style={thStyle}>{c.metric}</th>
                  <th style={thStyle}>{c.n}</th>
                  <th style={thStyle}>{c.meanCi}</th>
                  <th style={thStyle}>{c.interval}</th>
                  <th style={thStyle}>{c.std}</th>
                  <th style={thStyle}>{c.range}</th>
                </tr>
              </thead>
              <tbody>
                {ordered.map((name) => {
                  const agg = aggregates[name];
                  const shown = formatAggregate(agg);
                  const primary = name === group.metric;
                  return (
                    <tr key={name}>
                      <td
                        style={{
                          ...tdLabelStyle,
                          color: primary ? accent : "var(--vf-text-muted)",
                          fontWeight: primary ? 700 : 500,
                        }}
                      >
                        {name}
                      </td>
                      <td style={tdStyle}>{agg.n}</td>
                      <td style={tdStyle}>
                        <MeanCi agg={agg} accent={primary ? accent : undefined} />
                      </td>
                      <td style={tdStyle}>
                        {shown && shown.low !== null && shown.high !== null
                          ? `[${shown.low}, ${shown.high}]`
                          : t.runGroup.detail.noInterval}
                      </td>
                      <td style={tdStyle}>{shown?.std ?? "—"}</td>
                      <td style={tdStyle}>
                        {formatNumber(agg.min)} – {formatNumber(agg.max)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {/* A std is only comparable between runs that divide alike. */}
          <div style={noteStyle}>{t.runDetail.cv.stdNote(stdDdof({ std_ddof: group.std_ddof }))}</div>
          {/* Few folds is a K-fold's design, not a shortfall worth a warning. */}
          {!kfold && headline && hasFewSeeds(headline.n) && (
            <div style={{ ...noteStyle, color: STOPPED_COLOR }}>
              ⚠ {t.runGroup.detail.fewSeeds(headline.n)}
            </div>
          )}
        </GroupSection>
      )}
      <GroupSection title={kfold ? t.runGroup.detail.foldsTitle : t.runGroup.detail.seedsTitle}>
        <SeedsTable
          seeds={group.children ?? []}
          accent={accent}
          metric={group.metric}
          showVariant={false}
          byFold={kfold}
          knownRunIds={knownRunIds}
          onOpenRun={onOpenRun}
        />
      </GroupSection>
    </>
  );
}

function overridesText(variant: GroupVariant): string {
  const entries = Object.entries(variant.overrides);
  if (entries.length === 0) return "—";
  return entries.map(([key, value]) => `${key}=${String(value)}`).join(", ");
}

function TestsTable({ group }: { group: RunGroup }) {
  const t = useT();
  const c = t.runGroup.detail.cols;
  const tests: PairedTest[] = group.comparisons ?? [];
  const effects: Record<string, string> = t.runGroup.detail.effects;
  if (tests.length === 0) {
    return <div style={noteStyle}>{t.runGroup.detail.noTests}</div>;
  }
  return (
    <>
      <div style={tableBoxStyle}>
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>{c.pair}</th>
              <th style={thStyle}>{c.pairs}</th>
              <th style={thStyle}>{c.diff}</th>
              <th style={thStyle}>{c.test}</th>
              <th style={thStyle}>{c.p}</th>
              <th style={thStyle}>{c.holm}</th>
              <th style={thStyle}>{c.effect}</th>
            </tr>
          </thead>
          <tbody>
            {tests.map((test) => {
              const verdict = holmVerdict(test);
              return (
                <tr key={`${test.label_a}:${test.label_b}`}>
                  <td style={tdLabelStyle}>
                    {test.label_a} · {test.label_b}
                  </td>
                  <td style={tdStyle}>{test.n_pairs}</td>
                  <td style={tdStyle}>{formatNumber(test.mean_difference)}</td>
                  <td style={tdStyle} title={test.test_reason}>
                    {t.runGroup.detail.tests[test.test]}
                  </td>
                  <td style={tdStyle}>{formatP(test.p_value)}</td>
                  <td
                    style={{
                      ...tdStyle,
                      fontWeight: 700,
                      color: verdict === "yes" ? OK_COLOR : "var(--vf-text-dim)",
                    }}
                  >
                    {t.runGroup.detail[verdict]}
                    {test.underpowered && (
                      <span
                        title={t.runGroup.detail.underpoweredTitle(
                          test.n_pairs,
                          formatNumber(test.min_achievable_p),
                        )}
                        style={{
                          marginLeft: 8,
                          fontWeight: 400,
                          color: STOPPED_COLOR,
                          cursor: "help",
                        }}
                      >
                        ⚠ {t.runGroup.detail.underpowered}
                      </span>
                    )}
                  </td>
                  <td style={tdStyle}>
                    {formatNumber(test.effect_size, 2)}{" "}
                    <span style={{ color: "var(--vf-text-muted)" }}>
                      ({effects[test.effect_label] ?? test.effect_label})
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div style={noteStyle}>{t.runGroup.detail.testsNote}</div>
    </>
  );
}

function ComparisonBody({ group, accent, knownRunIds, onOpenRun }: RunGroupDetailProps) {
  const t = useT();
  const c = t.runGroup.detail.cols;
  const variants = group.variants ?? {};
  const best = bestByMean(group);
  const notRun = group.not_run ?? [];
  const skipped = group.skipped_variants ?? [];
  const allChildren = Object.values(variants).flatMap((v) => v.children);
  // Every variant has its own interval, so the caution is about the weakest one.
  const fewest = fewestSeeds(group);

  return (
    <>
      <GroupSection title={t.runGroup.detail.variantsTitle}>
        <div style={tableBoxStyle}>
          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={thStyle}>{c.variant}</th>
                <th style={thStyle}>{c.overrides}</th>
                <th style={thStyle}>{c.n}</th>
                <th style={thStyle}>{group.metric ?? c.metric}</th>
                <th style={thStyle}>{c.seeds}</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(variants).map(([label, variant]) => {
                const agg = group.metric ? variant.aggregates[group.metric] : undefined;
                const isBest = best.kind === "best" && best.label === label;
                return (
                  <tr key={label}>
                    <td
                      style={{
                        ...tdLabelStyle,
                        color: isBest ? accent : "var(--vf-text-muted)",
                        fontWeight: isBest ? 700 : 500,
                      }}
                    >
                      {label}
                    </td>
                    <td style={{ ...tdStyle, color: "var(--vf-text-dim)" }}>
                      {overridesText(variant)}
                    </td>
                    <td style={tdStyle}>{agg?.n ?? variant.successful ?? 0}</td>
                    <td style={tdStyle}>
                      <MeanCi agg={agg} accent={isBest ? accent : undefined} />
                    </td>
                    <td style={tdStyle}>
                      {variant.seeds_finished.length > 0 ? variant.seeds_finished.join(", ") : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <Row label={t.runGroup.detail.bestByMean}>
          {best.kind === "best" ? (
            <>
              {best.label}
              {(group.ranking_seeds?.length ?? 0) > 0 && (
                <span style={{ color: "var(--vf-text-muted)" }}>
                  {" · "}
                  {t.runGroup.detail.rankingSeeds((group.ranking_seeds ?? []).join(", "))}
                </span>
              )}
            </>
          ) : (
            <>
              —
              <span style={{ color: "var(--vf-text-muted)" }}>
                {" · "}
                {t.runGroup.detail.noBest[best.reason]}
              </span>
            </>
          )}
        </Row>
        <div style={noteStyle}>{t.runGroup.detail.bestNote}</div>
        {notRun.length > 0 && (
          <div style={{ ...noteStyle, color: STOPPED_COLOR }}>
            ■ {t.runGroup.detail.notRun(notRun.join(", "))}
          </div>
        )}
        {skipped.length > 0 && (
          <div style={{ ...noteStyle, color: STOPPED_COLOR }}>
            ⚠ {t.runGroup.detail.skipped(skipped.join(", "))}
          </div>
        )}
        <div style={noteStyle}>{t.runDetail.cv.stdNote(stdDdof({ std_ddof: group.std_ddof }))}</div>
        {fewest !== null && hasFewSeeds(fewest) && (
          <div style={{ ...noteStyle, color: STOPPED_COLOR }}>
            ⚠ {t.runGroup.detail.fewSeeds(fewest)}
          </div>
        )}
      </GroupSection>

      <GroupSection title={t.runGroup.detail.testsTitle}>
        <TestsTable group={group} />
      </GroupSection>

      <GroupSection title={t.runGroup.detail.seedsTitle}>
        <SeedsTable
          seeds={allChildren}
          accent={accent}
          metric={group.metric}
          showVariant
          knownRunIds={knownRunIds}
          onOpenRun={onOpenRun}
        />
      </GroupSection>
    </>
  );
}

/** A sweep or a model comparison (ADR-124): one row per trial, in the order
 *  the report kept, with what it tried and the trial its ranking put first. */
function SweepBody({ group, accent, knownRunIds, onOpenRun }: RunGroupDetailProps) {
  const t = useT();
  const c = t.runGroup.detail.cols;
  const metric = group.metric;
  return (
    <GroupSection title={t.runGroup.detail.trialsTitle}>
      <div style={tableBoxStyle}>
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>{c.trial}</th>
              <th style={thStyle}>{c.params}</th>
              <th style={thStyle}>{metric ?? c.metric}</th>
              <th style={thStyle}>{c.status}</th>
              <th style={thStyle}>{c.run}</th>
            </tr>
          </thead>
          <tbody>
            {(group.children ?? []).map((child) => {
              const best = child.trial_index != null && child.trial_index === group.best_index;
              return (
                <tr key={child.trial_index ?? child.run_id ?? ""}>
                  <td style={{ ...tdLabelStyle, color: best ? accent : undefined }}>
                    {child.trial_index != null ? child.trial_index + 1 : "—"}
                    {best && ` · ${t.runGroup.detail.best}`}
                  </td>
                  <td style={tdStyle}>{formatOverrides(child.overrides ?? {}) || "—"}</td>
                  <td style={tdStyle}>{formatNumber(metric ? child.metrics[metric] : null)}</td>
                  <td style={tdStyle}>
                    <ChildStatus child={child} />
                  </td>
                  <td style={tdStyle}>
                    <OpenRun child={child} accent={accent} knownRunIds={knownRunIds} onOpenRun={onOpenRun} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </GroupSection>
  );
}

/** What a replicate group shows in its run detail: the numbers its report
 *  recorded (mean ± CI95, the paired tests) and the runs it is made of. */
export function RunGroupDetail(props: RunGroupDetailProps) {
  const t = useT();
  const { group } = props;
  return (
    <>
      <GroupSection title={t.runGroup.detail.eyebrow[group.kind]}>
        <div>
          <Row label={t.runGroup.detail.metric}>
            {group.metric ?? "—"}
            {group.metric_direction && (
              <span style={{ color: "var(--vf-text-muted)" }}>
                {" · "}
                {t.runGroup.detail.direction[group.metric_direction]}
              </span>
            )}
          </Row>
          {group.kind === "kfold" ? (
            <Row label={t.runGroup.detail.foldsAsked}>{group.n_requested}</Row>
          ) : group.kind === "sweep" ? (
            <Row label={t.runGroup.detail.mode}>
              {t.runGroup.detail.modes[group.mode ?? ""] ?? group.mode ?? "—"}
            </Row>
          ) : (
            <Row label={t.runGroup.detail.seedsAsked}>
              {group.seeds.length > 0 ? group.seeds.join(", ") : "—"}
            </Row>
          )}
          <Row label={t.runGroup.detail.trainingsDone}>
            {group.n_finished}/{group.n_requested}
          </Row>
          {group.kind === "replicated_comparison" && typeof group.alpha === "number" && (
            <Row label={t.runGroup.detail.alpha}>{group.alpha}</Row>
          )}
          {group.report_path && <Row label={t.runGroup.detail.report}>{group.report_path}</Row>}
        </div>
        {group.stopped && (
          <div style={{ ...noteStyle, color: STOPPED_COLOR }}>■ {t.runGroup.detail.stopped}</div>
        )}
      </GroupSection>
      {group.kind === "replicated_comparison" ? (
        <ComparisonBody {...props} />
      ) : group.kind === "sweep" ? (
        <SweepBody {...props} />
      ) : (
        <ReplicatesBody {...props} />
      )}
    </>
  );
}
