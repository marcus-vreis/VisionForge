import { useCallback, useEffect, useState } from "react";
import {
  incorporateApply,
  incorporateCancel,
  incorporatePlan,
  incorporatePreview,
  incorporateStatus,
  pickDatasetFolder,
} from "../api/client";
import { SPLITS, type ExplorerForm } from "../lib/dataset-explorer";
import {
  buildPlanBody,
  isJobFinished,
  isValidDestName,
  jobPercent,
  parseSeed,
  planRows,
  ratioLabel,
  reasonRows,
  toggleClass,
  customRatios,
  type IncorporateJob,
  type IncorporatePlan,
  type IncorporatePreview,
  type IncorporateResult,
  type PercentTexts,
} from "../lib/dataset-incorporate";
import { runPicker } from "../lib/picker-feedback";
import { errorMessage as message, useJobPolling } from "../lib/useJobPolling";
import { useT } from "../i18n/useT";
import { Segmented, TextField } from "./controls";
import { button, card, ghostButton, mono, sectionLabel } from "./datasetExplorerStyles";
import { Progress } from "./JobProgress";

const danger = { ...mono, color: "var(--vf-danger, #f87171)" };

const pollPreview = (id: string) => incorporateStatus<IncorporatePreview>(id);
const pollApply = (id: string) => incorporateStatus<IncorporateResult>(id);

function PlanView({
  plan,
  preview,
  accepted,
  onToggle,
}: {
  plan: IncorporatePlan | null;
  preview: IncorporatePreview;
  accepted: string[];
  onToggle: (name: string) => void;
}) {
  const t = useT().datasetIncorporate;
  const ex = useT().datasetExplorer;
  const rows = plan ? planRows(plan) : [];
  const reasons = plan ? reasonRows(plan) : [];
  return (
    <>
      {preview.new_classes.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={sectionLabel}>{t.newClassesTitle}</div>
          <div style={mono}>{t.newClassesHelp}</div>
          {preview.new_classes.map((c) => (
            <label key={c.name} style={{ ...mono, display: "flex", gap: 6, alignItems: "center" }}>
              <input
                type="checkbox"
                checked={accepted.includes(c.name)}
                onChange={() => onToggle(c.name)}
              />
              {t.newClassItem(c.name, c.images)}
            </label>
          ))}
        </div>
      )}
      {plan && (
        <>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", ...mono }}>
            <span>{t.found(preview.source_images)}</span>
            <span>{t.willAdd(plan.added)}</span>
            <span>{t.willSkip(plan.excluded_count)}</span>
          </div>
          {plan.added === 0 ? (
            <div style={mono}>{t.nothingToAdd}</div>
          ) : (
            <>
              <div style={sectionLabel}>{t.addedPerClass}</div>
              <table style={{ ...mono, borderCollapse: "collapse", width: "100%" }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: "left" }}>{ex.classColumn}</th>
                    {SPLITS.filter((s) => preview.present[s]).map((s) => (
                      <th key={s} style={{ textAlign: "right" }}>
                        {ex.splitNames[s]}
                      </th>
                    ))}
                    <th style={{ textAlign: "right" }}>{ex.totalColumn}</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.name}>
                      <td>{r.name}</td>
                      {SPLITS.filter((s) => preview.present[s]).map((s) => (
                        <td key={s} style={{ textAlign: "right" }}>
                          {r.counts[s]}
                        </td>
                      ))}
                      <td style={{ textAlign: "right" }}>{r.total}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
          {plan.excluded_count > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <div style={sectionLabel}>{t.excludedTitle(plan.excluded_count)}</div>
              <div style={mono}>
                {reasons
                  .map((r) => t.reasonCount(t.reasons[r.reason] ?? r.reason, r.count))
                  .join("  ·  ")}
              </div>
              {plan.excluded.map((e) => (
                <div key={`${e.reason}:${e.path}`} style={{ ...mono, wordBreak: "break-all" }}>
                  {e.path} — {t.reasons[e.reason] ?? e.reason}
                  {e.detail ? ` (${e.detail})` : ""}
                </div>
              ))}
              {plan.excluded_count > plan.excluded.length && (
                <div style={mono}>{t.moreNotShown(plan.excluded_count - plan.excluded.length)}</div>
              )}
            </div>
          )}
        </>
      )}
    </>
  );
}

/** "Incorporar imagens" (ADR-166): choose a folder of new images, review the
 *  preview, name the new dataset and write it. The opened dataset never changes. */
export function IncorporatePanel({
  form,
  onOpenResult,
}: {
  form: ExplorerForm;
  /** Called with the folder of the new dataset so the explorer can open it. */
  onOpenResult: (path: string) => void;
}) {
  const t = useT().datasetIncorporate;
  const ex = useT().datasetExplorer;
  const [expanded, setExpanded] = useState(false);
  const [sourceDir, setSourceDir] = useState("");
  const [pickNote, setPickNote] = useState<string | null>(null);
  const [previewJob, setPreviewJob] = useState<IncorporateJob<IncorporatePreview> | null>(null);
  const [applyJob, setApplyJob] = useState<IncorporateJob<IncorporateResult> | null>(null);
  const [accepted, setAccepted] = useState<string[]>([]);
  const [custom, setCustom] = useState(false);
  const [percents, setPercents] = useState<PercentTexts>({ train: "70", val: "20", test: "10" });
  const [seedText, setSeedText] = useState("0");
  const [plan, setPlan] = useState<IncorporatePlan | null>(null);
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
  const onPreview = useCallback((j: IncorporateJob<IncorporatePreview>) => {
    setPreviewJob(j);
    if (isJobFinished(j.state)) setCancelling(false);
    if (j.state === "done" && j.result) setDestName((n) => n || j.result!.suggested_name);
  }, []);
  const onApply = useCallback((j: IncorporateJob<IncorporateResult>) => {
    setApplyJob(j);
    if (isJobFinished(j.state)) setCancelling(false);
  }, []);
  useJobPolling(previewJob, pollPreview, onPreview, fail);
  useJobPolling(applyJob, pollApply, onApply, fail);

  const present = preview?.present ?? { train: true, val: true, test: true };
  const body = buildPlanBody({ accepted, seedText, custom, percents, present });
  const bodyKey = JSON.stringify(body);

  // Re-plan whenever a choice changes; the preview itself is not recomputed.
  useEffect(() => {
    if (!previewId || !preview || body === null) return;
    let stale = false;
    const timer = setTimeout(() => {
      incorporatePlan(previewId, body)
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
    setAccepted([]);
    setDestName("");
    try {
      setPreviewJob(await incorporatePreview(form, sourceDir));
    } catch (e) {
      setError(message(e));
    }
  };

  const startApply = async () => {
    if (!previewId || body === null) return;
    setError(null);
    try {
      setApplyJob(await incorporateApply(previewId, body, destName));
    } catch (e) {
      setError(message(e));
    }
  };

  const cancel = async (jobId: string | undefined) => {
    if (!jobId) return;
    setCancelling(true);
    try {
      await incorporateCancel(jobId);
    } catch (e) {
      fail(message(e));
    }
  };

  const toggleCustom = (next: boolean) => {
    if (next && preview) {
      const total = SPLITS.reduce((n, s) => n + preview.base_counts[s], 0) || 1;
      setPercents({
        train: String(Math.round((preview.base_counts.train / total) * 100)),
        val: String(Math.round((preview.base_counts.val / total) * 100)),
        test: String(Math.round((preview.base_counts.test / total) * 100)),
      });
    }
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
    preview !== null && plan !== null && plan.added > 0 && body !== null && destOk && !applyRunning && !result;
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

      <div style={sectionLabel}>{t.sourceTitle}</div>
      <div style={mono}>{form.task === "detection" ? t.sourceDetection : t.sourceClassification}</div>
      <div style={{ display: "flex", alignItems: "flex-end", gap: 12 }}>
        <div style={{ flex: 1 }}>
          <TextField
            label={t.sourceDir}
            value={sourceDir}
            onChange={setSourceDir}
            placeholder={t.sourcePlaceholder}
            mono
          />
        </div>
        <button
          type="button"
          style={ghostButton}
          onClick={() =>
            void runPicker(
              pickDatasetFolder,
              (path) => {
                setSourceDir(path);
                setPickNote(null);
              },
              { canceled: t.pickCancelled, failed: t.pickFailed },
            ).then(setPickNote)
          }
        >
          {t.browse}
        </button>
        <button
          type="button"
          style={button}
          disabled={sourceDir.trim() === "" || previewRunning || applyRunning}
          onClick={() => void startPreview()}
        >
          {t.analyze}
        </button>
      </div>
      {pickNote && <div style={mono}>{pickNote}</div>}

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
          <div style={{ display: "flex", flexDirection: "column", gap: 8, maxWidth: 520 }}>
            <Segmented
              label={t.ratioTitle}
              value={custom ? "custom" : "same"}
              onChange={(v) => toggleCustom(v === "custom")}
              options={[
                {
                  value: "same",
                  label: plan
                    ? t.ratioSame(ratioLabel(plan.ratios, preview.present))
                    : t.ratioSame("…"),
                },
                { value: "custom", label: t.ratioCustom },
              ]}
            />
            {custom && (
              <div style={{ display: "flex", gap: 10 }}>
                {SPLITS.filter((s) => preview.present[s]).map((s) => (
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
            )}
            {custom && customRatios(percents, preview.present) === null && (
              <div style={danger}>{t.ratioInvalid}</div>
            )}
            {SPLITS.filter((s) => !preview.present[s]).map((s) => (
              <div key={s} style={mono}>
                {t.noSplit(ex.splitNames[s])}
              </div>
            ))}
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
          <PlanView
            plan={plan}
            preview={preview}
            accepted={accepted}
            onToggle={(name) => setAccepted((a) => toggleClass(a, name))}
          />
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
          <div style={mono}>{t.done(result.added, result.path)}</div>
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
