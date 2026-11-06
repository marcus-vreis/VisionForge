import { useCallback, useEffect, useState } from "react";
import {
  resplitApply,
  resplitCancel,
  resplitPlan,
  resplitPreview,
  resplitStatus,
} from "../api/client";
import { SPLITS, type ExplorerForm } from "../lib/dataset-explorer";
import {
  isJobFinished,
  isValidDestName,
  jobPercent,
  parseSeed,
  type IncorporateJob,
  type PercentTexts,
} from "../lib/dataset-incorporate";
import {
  buildBody,
  compareRows,
  currentPercents,
  customRatios,
  droppedReasonRows,
  effectiveLock,
  formatShare,
  ratioRows,
  vanishedSplits,
  type ResplitPlan,
  type ResplitPreview,
  type ResplitResult,
  type ResplitWarning,
} from "../lib/dataset-resplit";
import { errorMessage as message, useJobPolling } from "../lib/useJobPolling";
import { useT } from "../i18n/useT";
import { TextField } from "./controls";
import { button, card, ghostButton, mono, sectionLabel } from "./datasetExplorerStyles";
import { Progress } from "./JobProgress";

const danger = { ...mono, color: "var(--vf-danger, #f87171)" };

const pollPreview = (id: string) => resplitStatus<ResplitPreview>(id);
const pollApply = (id: string) => resplitStatus<ResplitResult>(id);

function WarningLine({ w }: { w: ResplitWarning }) {
  const t = useT().datasetResplit;
  const ex = useT().datasetExplorer;
  const split = w.split ? ex.splitNames[w.split] : "";
  const text =
    w.code === "new_split"
      ? t.warnNewSplit(split)
      : w.code === "empty_split"
        ? t.warnEmptySplit(split)
        : w.code === "split_removed"
          ? t.warnSplitRemoved(split)
          : w.code === "class_missing"
            ? t.warnClassMissing(w.class ?? "", split)
            : w.code;
  return <div style={mono}>{text}</div>;
}

function CompareTable({ plan }: { plan: ResplitPlan }) {
  const t = useT().datasetResplit;
  const ex = useT().datasetExplorer;
  const cell = (before: number, after: number) => (before === after ? `${after}` : `${before} → ${after}`);
  return (
    <table style={{ ...mono, borderCollapse: "collapse", width: "100%" }}>
      <thead>
        <tr>
          <th style={{ textAlign: "left" }}>{ex.classColumn}</th>
          {SPLITS.map((s) => (
            <th key={s} style={{ textAlign: "right" }}>
              {ex.splitNames[s]}
              {plan.locked_splits.includes(s) ? ` ${t.lockedMark}` : ""}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {compareRows(plan).map((r) => (
          <tr key={r.name}>
            <td>{r.name}</td>
            {SPLITS.map((s) => (
              <td key={s} style={{ textAlign: "right" }}>
                {cell(r.before[s], r.after[s])}
              </td>
            ))}
          </tr>
        ))}
        <tr>
          <td>{ex.totalColumn}</td>
          {SPLITS.map((s) => (
            <td key={s} style={{ textAlign: "right" }}>
              {cell(plan.before_totals[s], plan.after_totals[s])}
            </td>
          ))}
        </tr>
      </tbody>
    </table>
  );
}

function PlanView({ plan, preview }: { plan: ResplitPlan; preview: ResplitPreview }) {
  const t = useT().datasetResplit;
  const ex = useT().datasetExplorer;
  const reasons = droppedReasonRows(plan);
  const vanished = vanishedSplits(plan, preview.present);
  return (
    <>
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", ...mono }}>
        <span>{t.keptLine(plan.kept)}</span>
        <span>{t.movedLine(plan.moved)}</span>
        <span>{t.droppedLine(plan.dropped_count)}</span>
      </div>
      <div style={sectionLabel}>{t.tableTitle}</div>
      <CompareTable plan={plan} />
      <div style={sectionLabel}>{t.ratiosTitle}</div>
      {ratioRows(plan).map((r) => (
        <div key={r.split} style={mono}>
          {t.ratioLine(ex.splitNames[r.split], formatShare(r.requested), formatShare(r.achieved))}
        </div>
      ))}
      {plan.duplicate_sets > 0 && (
        <div style={mono}>
          {t.duplicatesLine(plan.duplicate_sets, plan.duplicate_extras)}
          {plan.duplicate_sets_split_before > 0 &&
            ` ${t.duplicatesLeak(plan.duplicate_sets_split_before)}`}
        </div>
      )}
      {vanished.map((s) => (
        <div key={s} style={danger}>
          {t.warnSplitEmptyResult(ex.splitNames[s])}
        </div>
      ))}
      {plan.warnings.map((w) => (
        <WarningLine key={`${w.code}:${w.split ?? ""}:${w.class ?? ""}`} w={w} />
      ))}
      {plan.dropped_count > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={sectionLabel}>{t.droppedTitle(plan.dropped_count)}</div>
          <div style={mono}>{t.droppedHelp}</div>
          <div style={mono}>
            {reasons
              .map((r) => t.reasonCount(t.reasons[r.reason] ?? r.reason, r.count))
              .join("  ·  ")}
          </div>
          {plan.dropped.map((d) => (
            <div key={d.path} style={{ ...mono, wordBreak: "break-all" }}>
              {d.path} — {t.reasons[d.reason] ?? d.reason}
            </div>
          ))}
          {plan.dropped_count > plan.dropped.length && (
            <div style={mono}>{t.moreNotShown(plan.dropped_count - plan.dropped.length)}</div>
          )}
        </div>
      )}
    </>
  );
}

/** "Refazer divisão" (ADR-168): pool the train and validation images (the test
 *  split stays as it is by default), redistribute them by the chosen ratios and
 *  write a new dataset. The opened dataset never changes. */
export function ResplitPanel({
  form,
  onOpenResult,
}: {
  form: ExplorerForm;
  /** Called with the folder of the new dataset so the explorer can open it. */
  onOpenResult: (path: string) => void;
}) {
  const t = useT().datasetResplit;
  const ex = useT().datasetExplorer;
  const [expanded, setExpanded] = useState(false);
  const [groupCsv, setGroupCsv] = useState("");
  const [previewJob, setPreviewJob] = useState<IncorporateJob<ResplitPreview> | null>(null);
  const [applyJob, setApplyJob] = useState<IncorporateJob<ResplitResult> | null>(null);
  const [lockTest, setLockTest] = useState(true);
  const [byGroup, setByGroup] = useState(true);
  const [custom, setCustom] = useState(false);
  const [percents, setPercents] = useState<PercentTexts>({ train: "80", val: "20", test: "0" });
  const [seedText, setSeedText] = useState("0");
  const [plan, setPlan] = useState<ResplitPlan | null>(null);
  const [destName, setDestName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);

  const preview = previewJob?.state === "done" ? previewJob.result : null;
  const previewRunning = previewJob !== null && !isJobFinished(previewJob.state);
  const applyRunning = applyJob !== null && !isJobFinished(applyJob.state);
  const result = applyJob?.state === "done" ? applyJob.result : null;
  const previewId = previewJob?.job_id ?? null;

  const fail = useCallback((m: string) => {
    setError(m);
    setCancelling(false);
  }, []);
  const onPreview = useCallback((j: IncorporateJob<ResplitPreview>) => {
    setPreviewJob(j);
    if (isJobFinished(j.state)) setCancelling(false);
    if (j.state === "done" && j.result) setDestName((n) => n || j.result!.suggested_name);
  }, []);
  const onApply = useCallback((j: IncorporateJob<ResplitResult>) => {
    setApplyJob(j);
    if (isJobFinished(j.state)) setCancelling(false);
  }, []);
  useJobPolling(previewJob, pollPreview, onPreview, fail);
  useJobPolling(applyJob, pollApply, onApply, fail);

  const present = preview?.present ?? { train: true, val: true, test: true };
  const hasGroups = preview !== null && preview.group_count !== null;
  const locked = effectiveLock(lockTest, present);
  const body = buildBody({
    seedText,
    lockTest,
    present,
    custom,
    percents,
    byGroup: hasGroups ? byGroup : null,
  });
  const bodyKey = JSON.stringify(body);

  // Re-plan whenever a choice changes; the preview itself is not recomputed.
  useEffect(() => {
    if (!previewId || !preview || body === null) return;
    let stale = false;
    const timer = setTimeout(() => {
      resplitPlan(previewId, body)
        .then((next) => {
          if (stale) return;
          setPlan(next);
          setError(null);
        })
        .catch((e: unknown) => {
          if (!stale) setError(message(e));
        });
    }, 250);
    return () => {
      stale = true;
      clearTimeout(timer);
    };
    // `body` is covered by `bodyKey`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewId, preview, bodyKey]);

  const startPreview = async () => {
    setError(null);
    setPreviewJob(null);
    setApplyJob(null);
    setPlan(null);
    setDestName("");
    try {
      setPreviewJob(await resplitPreview(form, groupCsv));
    } catch (e) {
      setError(message(e));
    }
  };

  const startApply = async () => {
    if (!previewId || body === null) return;
    setError(null);
    try {
      setApplyJob(await resplitApply(previewId, body, destName));
    } catch (e) {
      setError(message(e));
    }
  };

  const cancel = async (jobId: string | undefined) => {
    if (!jobId) return;
    setCancelling(true);
    try {
      await resplitCancel(jobId);
    } catch (e) {
      fail(message(e));
    }
  };

  const toggleCustom = (next: boolean) => {
    if (next && preview) setPercents(currentPercents(preview.per_split, locked));
    setCustom(next);
  };

  if (!expanded) {
    return (
      <div>
        <button type="button" style={button} onClick={() => setExpanded(true)}>
          {t.open}
        </button>
      </div>
    );
  }

  const destOk = isValidDestName(destName);
  const canApply =
    preview !== null &&
    plan !== null &&
    plan.kept > 0 &&
    body !== null &&
    destOk &&
    !applyRunning &&
    !result;
  const workingPct = previewRunning
    ? jobPercent(previewJob?.done ?? 0, previewJob?.total ?? 0)
    : jobPercent(applyJob?.done ?? 0, applyJob?.total ?? 0);

  return (
    <div style={{ ...card, flexBasis: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={sectionLabel}>{t.open}</div>
        <button type="button" style={ghostButton} onClick={() => setExpanded(false)}>
          {t.close}
        </button>
      </div>
      <div style={mono}>{t.intro}</div>

      <div style={sectionLabel}>{t.groupTitle}</div>
      <div style={mono}>{t.groupHelp}</div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 12 }}>
        <div style={{ flex: 1 }}>
          <TextField
            label={t.groupFile}
            value={groupCsv}
            onChange={setGroupCsv}
            placeholder={t.groupPlaceholder}
            mono
          />
        </div>
        <button
          type="button"
          style={button}
          disabled={previewRunning || applyRunning}
          onClick={() => void startPreview()}
        >
          {t.analyze}
        </button>
      </div>

      {(previewRunning || applyRunning) && (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={mono}>
            {previewRunning
              ? t.analyzing(previewJob?.done ?? 0, previewJob?.total ?? 0)
              : t.applying(applyJob?.done ?? 0, applyJob?.total ?? 0)}
          </div>
          <Progress pct={workingPct} />
          <div>
            <button
              type="button"
              style={ghostButton}
              disabled={cancelling}
              onClick={() => void cancel(previewRunning ? previewJob?.job_id : applyJob?.job_id)}
            >
              {cancelling ? t.cancelling : t.cancel}
            </button>
          </div>
        </div>
      )}
      {previewJob?.state === "cancelled" && <div style={mono}>{t.cancelled}</div>}
      {applyJob?.state === "cancelled" && <div style={mono}>{t.cancelled}</div>}
      {previewJob?.state === "failed" && <div style={danger}>{t.failed(previewJob.error ?? "")}</div>}
      {applyJob?.state === "failed" && <div style={danger}>{t.failed(applyJob.error ?? "")}</div>}
      {error && <div style={danger}>{t.failed(error)}</div>}

      {preview && (
        <>
          <div style={sectionLabel}>{t.previewTitle}</div>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", ...mono }}>
            <span>{t.found(preview.images)}</span>
            {preview.group_count !== null && (
              <span>{t.groupsInfo(preview.group_count, preview.ungrouped)}</span>
            )}
            {preview.unknown_entries > 0 && <span>{t.unknownEntries(preview.unknown_entries)}</span>}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8, maxWidth: 560 }}>
            <label style={{ ...mono, display: "flex", gap: 6, alignItems: "center" }}>
              <input
                type="checkbox"
                checked={locked}
                disabled={!preview.present.test}
                onChange={(e) => setLockTest(e.target.checked)}
              />
              {preview.present.test ? t.lockTest : t.lockNoTest}
            </label>
            {preview.present.test && <div style={mono}>{locked ? t.lockOnHelp : t.lockOffHelp}</div>}
            {hasGroups && (
              <label style={{ ...mono, display: "flex", gap: 6, alignItems: "center" }}>
                <input
                  type="checkbox"
                  checked={byGroup}
                  onChange={(e) => setByGroup(e.target.checked)}
                />
                {t.byGroup}
              </label>
            )}
            <label style={{ ...mono, display: "flex", gap: 6, alignItems: "center" }}>
              <input
                type="checkbox"
                checked={custom}
                onChange={(e) => toggleCustom(e.target.checked)}
              />
              {t.ratioCustom}
            </label>
            {!custom && <div style={mono}>{t.ratioSame}</div>}
            {custom && (
              <>
                <div style={{ display: "flex", gap: 10 }}>
                  {SPLITS.filter((s) => !(locked && s === "test")).map((s) => (
                    <div key={s} style={{ flex: 1 }}>
                      <TextField
                        label={t.ratioPercent(ex.splitNames[s])}
                        value={percents[s]}
                        onChange={(v) => setPercents((p) => ({ ...p, [s]: v }))}
                        mono
                      />
                    </div>
                  ))}
                </div>
                <div style={mono}>{t.ratioHelp}</div>
                {customRatios(percents, locked) === null && (
                  <div style={danger}>{t.ratioInvalid}</div>
                )}
              </>
            )}
            <div style={{ width: 180 }}>
              <TextField
                label={t.seed}
                value={seedText}
                onChange={setSeedText}
                hint={t.seedHint}
                mono
                error={parseSeed(seedText) === null ? t.seedInvalid : undefined}
              />
            </div>
          </div>
          {plan && <PlanView plan={plan} preview={preview} />}
          <div style={sectionLabel}>{t.destTitle}</div>
          <div style={{ display: "flex", alignItems: "flex-end", gap: 12 }}>
            <div style={{ flex: 1, maxWidth: 420 }}>
              <TextField
                label={t.destName}
                value={destName}
                onChange={setDestName}
                hint={t.destHint}
                mono
                error={destName !== "" && !destOk ? t.destInvalid : undefined}
              />
            </div>
            <button type="button" style={button} disabled={!canApply} onClick={() => void startApply()}>
              {t.apply}
            </button>
          </div>
        </>
      )}

      {result && (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={sectionLabel}>{t.doneTitle}</div>
          <div style={mono}>{t.done(result.images, result.moved, result.path)}</div>
          <div style={mono}>{t.linkedCopied(result.files_linked, result.files_copied)}</div>
          <div>
            <button
              type="button"
              style={button}
              onClick={() => {
                onOpenResult(result.path);
                setExpanded(false);
              }}
            >
              {t.openResult}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
