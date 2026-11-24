import { useCallback, useEffect, useState } from "react";
import { FolderOpen } from "lucide-react";
import {
  mergeApply,
  mergeCancel,
  mergePlan,
  mergePreview,
  mergeStatus,
  pickDatasetFolder,
} from "../api/client";
import { SPLITS, type ExplorerForm } from "../lib/dataset-explorer";
import { isJobFinished, isValidDestName, jobPercent, type IncorporateJob } from "../lib/dataset-incorporate";
import {
  SIDES,
  activeHints,
  buildMapping,
  canMerge,
  finalOrder,
  initialRows,
  mergedTargets,
  presentSplits,
  reasonRows,
  rowKey,
  rowProblems,
  splitsFromOne,
  targetSuggestions,
  updateRow,
  type MappingBody,
  type MergePlan,
  type MergePreview,
  type MergeResult,
  type MergeRow,
  type Side,
  type TargetProblem,
} from "../lib/dataset-merge";
import { runPicker } from "../lib/picker-feedback";
import { errorMessage, useJobPolling } from "../lib/useJobPolling";
import { useT } from "../i18n/useT";
import { ROUTE, useRouteOpen } from "../lib/session";
import { useDestHint } from "../lib/dest-hint";
import { ServerFoldersNote } from "./ServerFoldersNote";
import { TextField } from "./controls";
import { danger, ghostButton, mono, sectionLabel, stepTitle, text, warn } from "./datasetExplorerStyles";
import { ActionRow, AnalyzeButton, ApplyButton, DataTable, LabStep, LabTool, LearnMore } from "./DatasetLabParts";
import { Progress } from "./JobProgress";
import { ActionButton } from "./ui/ActionButton";
const DATALIST_ID = "vf-merge-targets";
const cell = { padding: "3px 8px 3px 0", verticalAlign: "middle" } as const;
const right = { ...cell, textAlign: "right" } as const;
const targetInput = {
  ...mono,
  width: "100%",
  minWidth: 120,
  padding: "4px 8px",
  background: "transparent",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 6,
  color: "var(--vf-text)",
} as const;

const pollPreview = (id: string) => mergeStatus<MergePreview>(id);
const pollApply = (id: string) => mergeStatus<MergeResult>(id);

function MappingTable({
  preview,
  rows,
  problems,
  onChange,
}: {
  preview: MergePreview;
  rows: MergeRow[];
  problems: Map<string, TargetProblem>;
  onChange: (side: Side, name: string, patch: Partial<Pick<MergeRow, "target" | "discard">>) => void;
}) {
  const t = useT().datasetMerge;
  const merged = mergedTargets(rows);
  const origin: Record<Side, string> = { a: t.sideA, b: t.sideB };
  return (
    <>
      <DataTable>
        <thead>
          <tr>
            <th style={{ textAlign: "left" }}>{t.colOrigin}</th>
            <th style={{ textAlign: "left" }}>{t.colClass}</th>
            <th style={{ textAlign: "right" }}>{t.colImages}</th>
            <th style={{ textAlign: "left" }}>{t.colTarget}</th>
            <th style={{ textAlign: "left" }}>{t.colDiscard}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const problem = problems.get(rowKey(r));
            return (
              <tr key={rowKey(r)} style={r.discard ? { opacity: 0.55 } : undefined}>
                <td style={cell}>{origin[r.side]}</td>
                <td style={cell}>{r.name}</td>
                <td style={right}>{r.images}</td>
                <td style={cell}>
                  <input
                    list={DATALIST_ID}
                    value={r.target}
                    disabled={r.discard}
                    aria-label={t.targetLabel(r.name)}
                    aria-invalid={problem !== undefined}
                    style={targetInput}
                    onChange={(e) => onChange(r.side, r.name, { target: e.target.value })}
                  />
                  {problem && <div style={danger}>{t.problems[problem]}</div>}
                  {!r.discard && !problem && merged.has(r.target.trim()) && <span style={mono}>{t.merged}</span>}
                </td>
                <td style={cell}>
                  <input
                    type="checkbox"
                    checked={r.discard}
                    aria-label={t.discardLabel(r.name)}
                    onChange={(e) => onChange(r.side, r.name, { discard: e.target.checked })}
                  />
                </td>
              </tr>
            );
          })}
        </tbody>
      </DataTable>
      <datalist id={DATALIST_ID}>
        {targetSuggestions(preview).map((n) => (
          <option key={n} value={n} />
        ))}
      </datalist>
    </>
  );
}

function DroppedView({ plan }: { plan: MergePlan }) {
  const t = useT().datasetMerge;
  const side: Record<Side, string> = { a: t.sideA, b: t.sideB };
  if (plan.dropped_count === 0 && plan.duplicates_dropped === 0) return null;
  return (
    <>
      {plan.dropped_count > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={sectionLabel}>{t.droppedTitle(plan.dropped_count)}</div>
          <div style={mono}>
            {reasonRows(plan)
              .map((r) => t.reasonCount(t.reasons[r.reason] ?? r.reason, r.count))
              .join("  ·  ")}
          </div>
          {plan.dropped.map((d) => (
            <div key={`${d.side}:${d.path}`} style={{ ...mono, wordBreak: "break-all" }}>
              {t.droppedItem(side[d.side], d.path, t.reasons[d.reason] ?? d.reason)}
            </div>
          ))}
          {plan.dropped_count > plan.dropped.length && (
            <div style={mono}>{t.moreNotShown(plan.dropped_count - plan.dropped.length)}</div>
          )}
        </div>
      )}
      {plan.duplicates_dropped > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={sectionLabel}>{t.duplicatesTitle(plan.duplicates_dropped)}</div>
          <div style={text}>{t.duplicatesHelp}</div>
          {plan.duplicates_cross_split > 0 && <div style={mono}>{t.leakTitle(plan.duplicates_cross_split)}</div>}
          {plan.duplicates.map((d) => (
            <div key={d.b_path} style={{ ...mono, wordBreak: "break-all" }}>
              {t.duplicateItem(d.b_path, d.a_path)}
            </div>
          ))}
          {plan.duplicates_dropped > plan.duplicates.length && (
            <div style={mono}>{t.moreNotShown(plan.duplicates_dropped - plan.duplicates.length)}</div>
          )}
        </div>
      )}
    </>
  );
}

function PlanView({ plan, detection }: { plan: MergePlan; detection: boolean }) {
  const t = useT().datasetMerge;
  const ex = useT().datasetExplorer;
  const splits = presentSplits(plan);
  const side: Record<Side, string> = { a: t.sideA, b: t.sideB };
  if (plan.images_total === 0) return <div style={warn}>{t.nothingLeft}</div>;
  return (
    <>
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", ...mono }}>
        <span>{t.totalImages(plan.images_total)}</span>
        {detection && SPLITS.some((s) => plan.background[s] > 0) && (
          <span>{t.background(SPLITS.reduce((n, s) => n + plan.background[s], 0))}</span>
        )}
      </div>
      <div style={sectionLabel}>{t.classesTitle}</div>
      <DataTable>
        <thead>
          <tr>
            <th style={{ textAlign: "left" }}>{ex.classColumn}</th>
            {splits.map((s) => (
              <th key={s} style={{ textAlign: "right" }}>
                {ex.splitNames[s]}
              </th>
            ))}
            <th style={{ textAlign: "right" }}>{ex.totalColumn}</th>
            {detection && <th style={{ textAlign: "right" }}>{t.boxesColumn}</th>}
          </tr>
        </thead>
        <tbody>
          {plan.classes.map((c) => (
            <tr key={c.name}>
              <td style={cell}>{c.name}</td>
              {splits.map((s) => (
                <td key={s} style={right}>
                  {c.images[s]}
                </td>
              ))}
              <td style={right}>{c.images.total}</td>
              {detection && <td style={right}>{c.boxes}</td>}
            </tr>
          ))}
        </tbody>
      </DataTable>
      <div style={sectionLabel}>{t.splitsTitle}</div>
      {splits.map((s) => (
        <div key={s} style={mono}>
          {ex.splitNames[s]}: {plan.splits[s].images} — {t.splitCounts(plan.splits[s].from_a, plan.splits[s].from_b)}
        </div>
      ))}
      {splitsFromOne(plan).map(([from, s]) => (
        <div key={`${from}:${s}`} style={mono}>
          {t.splitOnlyIn(ex.splitNames[s], side[from])}
        </div>
      ))}
    </>
  );
}

/** "Juntar com outro dataset" (ADR-167): pick the other dataset, map its classes
 *  onto the opened one's (rename, merge, discard), review the result and write a
 *  new dataset. Neither source ever changes. */
export function MergePanel({
  form,
  onOpenResult,
}: {
  form: ExplorerForm;
  /** Called with the folder of the new dataset so the explorer can open it. */
  onOpenResult: (path: string) => void;
}) {
  const t = useT().datasetMerge;
  const lab = useT().datasetLab;
  const canPick = useRouteOpen(ROUTE.pickFolder);
  const destHint = useDestHint(t.destHint);
  const [otherDir, setOtherDir] = useState("");
  const [pickNote, setPickNote] = useState<string | null>(null);
  const [previewJob, setPreviewJob] = useState<IncorporateJob<MergePreview> | null>(null);
  const [applyJob, setApplyJob] = useState<IncorporateJob<MergeResult> | null>(null);
  const [rows, setRows] = useState<MergeRow[]>([]);
  const [plan, setPlan] = useState<MergePlan | null>(null);
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
  const onPreview = useCallback((j: IncorporateJob<MergePreview>) => {
    setPreviewJob(j);
    if (isJobFinished(j.state)) setCancelling(false);
    if (j.state === "done" && j.result) {
      setRows(initialRows(j.result));
      setDestName((n) => n || j.result!.suggested_name);
    }
  }, []);
  const onApply = useCallback((j: IncorporateJob<MergeResult>) => {
    setApplyJob(j);
    if (isJobFinished(j.state)) setCancelling(false);
  }, []);
  useJobPolling(previewJob, pollPreview, onPreview, fail);
  useJobPolling(applyJob, pollApply, onApply, fail);

  const task = preview?.task ?? form.task;
  const problems = rowProblems(rows, task);
  const mappingKey = JSON.stringify(buildMapping(rows));
  const valid = problems.size === 0;

  // Re-plan whenever the mapping changes; the preview itself is not recomputed.
  useEffect(() => {
    if (!previewId || !preview || !valid) return;
    let stale = false;
    const timer = setTimeout(() => {
      mergePlan(previewId, JSON.parse(mappingKey) as MappingBody)
        .then((next) => {
          if (stale) return;
          setPlan(next);
          setError(null);
        })
        .catch((e: unknown) => {
          if (stale) return;
          setPlan(null);
          setError(errorMessage(e));
        });
    }, 250);
    return () => {
      stale = true;
      clearTimeout(timer);
    };
  }, [previewId, preview, valid, mappingKey]);

  const startPreview = async () => {
    setError(null);
    setPreviewJob(null);
    setApplyJob(null);
    setPlan(null);
    setRows([]);
    setDestName("");
    try {
      setPreviewJob(await mergePreview(form, otherDir));
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  const startApply = async () => {
    if (!previewId) return;
    setError(null);
    try {
      setApplyJob(await mergeApply(previewId, buildMapping(rows), destName));
    } catch (e) {
      setError(errorMessage(e));
    }
  };

  const cancel = async (jobId: string | undefined) => {
    if (!jobId) return;
    setCancelling(true);
    try {
      await mergeCancel(jobId);
    } catch (e) {
      fail(errorMessage(e));
    }
  };

  const destOk = isValidDestName(destName);
  const canApply = preview !== null && canMerge(problems, plan) && destOk && !applyRunning && !result;
  const workingPct = previewRunning
    ? jobPercent(previewJob?.done ?? 0, previewJob?.total ?? 0)
    : jobPercent(applyJob?.done ?? 0, applyJob?.total ?? 0);
  const hints = preview ? activeHints(preview, rows) : [];

  return (
    <LabTool title={t.open} lead={lab.leadMerge} more={t.intro}>
      <LabStep title={t.sourceTitle}>
        <div style={text}>{t.sourceHelp}</div>
        <ActionRow
          actions={
            <>
              <button
                type="button"
                hidden={!canPick}
                style={ghostButton}
                onClick={() =>
                  void runPicker(
                    pickDatasetFolder,
                    (path) => {
                      setOtherDir(path);
                      setPickNote(null);
                    },
                    { canceled: t.pickCancelled, failed: t.pickFailed },
                  ).then(setPickNote)
                }
              >
                {t.browse}
              </button>
              <AnalyzeButton
                label={t.analyze}
                disabled={otherDir.trim() === "" || previewRunning || applyRunning}
                onClick={() => void startPreview()}
              />
            </>
          }
        >
          <TextField
            label={t.sourceDir}
            value={otherDir}
            onChange={setOtherDir}
            placeholder={t.sourcePlaceholder}
            mono
          />
        </ActionRow>
        <ServerFoldersNote />
        {pickNote && <div style={text}>{pickNote}</div>}
      </LabStep>

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
      {previewJob?.state === "cancelled" && <div style={text}>{t.cancelled}</div>}
      {applyJob?.state === "cancelled" && <div style={text}>{t.cancelled}</div>}
      {previewJob?.state === "failed" && <div style={danger}>{t.failed(previewJob.error ?? "")}</div>}
      {applyJob?.state === "failed" && <div style={danger}>{t.failed(applyJob.error ?? "")}</div>}
      {error && <div style={danger}>{t.failed(error)}</div>}

      {preview && (
        <>
          <LabStep title={t.mappingTitle}>
            <div style={mono}>{t.found(preview.found.a, preview.found.b)}</div>
            {SIDES.some((s) => preview.repeated_within[s] > 0) && (
              <div style={text}>{t.repeatedWithin(preview.repeated_within.a, preview.repeated_within.b)}</div>
            )}
            <LearnMore summary={lab.mappingMore}>{t.mappingHelp}</LearnMore>
            {hints.map((group) => (
              <div key={group.join("|")} style={warn}>
                {t.similarHint(group.join(", "))}
              </div>
            ))}
            <MappingTable
              preview={preview}
              rows={rows}
              problems={problems}
              onChange={(side, name, patch) => setRows((r) => updateRow(r, side, name, patch))}
            />
            <div style={mono}>{finalOrder(rows).join("  ›  ")}</div>
          </LabStep>
          <LabStep title={t.resultTitle}>
            {plan && <PlanView plan={plan} detection={preview.task === "detection"} />}
            {plan && <DroppedView plan={plan} />}
          </LabStep>
          <LabStep title={t.destTitle}>
            <ActionRow
              actions={
                <ApplyButton label={t.apply} disabled={!canApply} onClick={() => void startApply()} />
              }
            >
              <TextField
                label={t.destName}
                value={destName}
                onChange={setDestName}
                hint={destHint}
                mono
                error={destName !== "" && !destOk ? t.destInvalid : undefined}
              />
            </ActionRow>
          </LabStep>
        </>
      )}

      {result && (
        <div className="vf-lab-done">
          <h3 style={{ ...stepTitle, margin: 0 }}>{t.doneTitle}</h3>
          <div style={mono}>{t.done(result.images, result.path)}</div>
          <div style={mono}>{t.linkedCopied(result.files_linked, result.files_copied)}</div>
          <div>
            <ActionButton icon={<FolderOpen />} onClick={() => onOpenResult(result.path)}>
              {t.openResult}
            </ActionButton>
          </div>
        </div>
      )}
    </LabTool>
  );
}
