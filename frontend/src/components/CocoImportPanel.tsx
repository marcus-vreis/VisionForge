import { useCallback, useEffect, useState } from "react";
import { FolderOpen } from "lucide-react";
import {
  cocoImportApply,
  cocoImportCancel,
  cocoImportPlan,
  cocoImportPreview,
  cocoImportStatus,
  pickDatasetFolder,
} from "../api/client";
import { SPLITS } from "../lib/dataset-explorer";
import {
  isJobFinished,
  isValidDestName,
  jobPercent,
  type IncorporateJob,
} from "../lib/dataset-incorporate";
import {
  SKIP,
  canApplyImport,
  initialChoices,
  reasonTotal,
  splitConflicts,
  splitsBody,
  undecided,
  type CocoImportPlan,
  type CocoImportPreview,
  type CocoImportResult,
  type CocoImportSource,
  type SplitChoice,
} from "../lib/dataset-coco";
import { runPicker } from "../lib/picker-feedback";
import { errorMessage as message, useJobPolling } from "../lib/useJobPolling";
import { useT } from "../i18n/useT";
import { ROUTE, useRouteOpen } from "../lib/session";
import { ServerFoldersNote } from "./ServerFoldersNote";
import { CocoJobBar, DestinationRow, ReasonSummary } from "./CocoParts";
import { TextField } from "./controls";
import {
  card,
  danger,
  ghostButton,
  mono,
  sectionLabel,
  stepTitle,
  text,
  warn,
} from "./datasetExplorerStyles";
import { ActionRow, AnalyzeButton, DataTable, LabStep, LabTool } from "./DatasetLabParts";
import { ActionButton } from "./ui/ActionButton";

const pollPreview = (id: string) => cocoImportStatus<CocoImportPreview>(id);
const pollApply = (id: string) => cocoImportStatus<CocoImportResult>(id);

const selectStyle = {
  ...mono,
  padding: "4px 8px",
  background: "transparent",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 6,
  color: "var(--vf-text)",
} as const;

function SourceCard({
  source,
  choice,
  onChoice,
}: {
  source: CocoImportSource;
  choice: SplitChoice;
  onChoice: (c: SplitChoice) => void;
}) {
  const t = useT().datasetCoco;
  const ex = useT().datasetExplorer;
  const selectId = `vf-coco-split-${source.name}`;
  return (
    <div style={{ ...card, padding: 12, gap: 6 }}>
      <div style={{ ...mono, wordBreak: "break-all" }}>
        {t.jsonFile(source.name)}
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          flexWrap: "wrap",
        }}
      >
        <label htmlFor={selectId} style={mono}>
          {t.splitFor(source.name)}
        </label>
        <select
          id={selectId}
          style={selectStyle}
          value={choice}
          onChange={(e) => onChoice(e.target.value as SplitChoice)}
        >
          <option value="">{t.splitUnset}</option>
          {SPLITS.map((s) => (
            <option key={s} value={s}>
              {ex.splitNames[s]}
            </option>
          ))}
          <option value={SKIP}>{t.splitSkip}</option>
        </select>
      </div>
      <div style={source.detected_split ? text : warn}>
        {source.detected_split
          ? t.detectedHint(ex.splitNames[source.detected_split])
          : t.undetectedHint}
      </div>
      <div style={source.images_dir === null ? warn : mono}>
        {source.images_dir === null
          ? t.imagesFolderMissing
          : t.imagesFolder(source.images_dir)}
      </div>
      <div style={mono}>
        {t.sourceCounts(
          source.images_listed,
          source.images_kept,
          source.annotations,
          source.boxes,
        )}
        {"; "}
        {t.categoriesLine(source.categories)}
      </div>
      {reasonTotal(source.skipped_by_reason) > 0 && (
        <>
          <div style={sectionLabel}>{t.skippedTitle}</div>
          <ReasonSummary
            counts={source.skipped_by_reason}
            labels={t.importReasons}
          />
        </>
      )}
      {reasonTotal(source.adjusted_by_reason) > 0 && (
        <>
          <div style={sectionLabel}>{t.adjustedTitle}</div>
          <ReasonSummary
            counts={source.adjusted_by_reason}
            labels={t.importReasons}
          />
        </>
      )}
      {source.listed.map((l) => (
        <div
          key={`${l.reason}:${l.item}`}
          style={{ ...mono, wordBreak: "break-all" }}
        >
          {l.item} — {t.importReasons[l.reason] ?? l.reason}
        </div>
      ))}
    </div>
  );
}

function PlanView({ plan }: { plan: CocoImportPlan }) {
  const t = useT().datasetCoco;
  const ex = useT().datasetExplorer;
  return (
    <>
      <div style={mono}>{t.planLine(plan.images, plan.boxes)}</div>
      <DataTable maxWidth={420}>
        <thead>
          <tr>
            <th style={{ textAlign: "left" }}>{t.splitColumn}</th>
            <th style={{ textAlign: "right" }}>{t.imagesColumn}</th>
            <th style={{ textAlign: "right" }}>{t.boxesColumn}</th>
          </tr>
        </thead>
        <tbody>
          {SPLITS.filter((s) => plan.per_split[s].images > 0).map((s) => (
            <tr key={s}>
              <td>{ex.splitNames[s]}</td>
              <td style={{ textAlign: "right" }}>{plan.per_split[s].images}</td>
              <td style={{ textAlign: "right" }}>{plan.per_split[s].boxes}</td>
            </tr>
          ))}
        </tbody>
      </DataTable>
      <div style={sectionLabel}>{t.mappingTitle}</div>
      {plan.mapping.map((m) => (
        <div key={m.coco_id} style={mono}>
          {t.mappingLine(m.coco_id, m.yolo_id, m.name)}
          {m.name !== m.original_name && ` ${t.renamedMark(m.original_name)}`}
        </div>
      ))}
    </>
  );
}

/** "Importar COCO" (ADR-169): read a COCO folder the user picks and write a new YOLO
 *  detection dataset next to it. Always available, with or without a dataset open. */
export function CocoImportPanel({
  onOpenResult,
}: {
  /** Called with the folder of the new YOLO dataset so the explorer can open it. */
  onOpenResult: (path: string) => void;
}) {
  const t = useT().datasetCoco;
  const ex = useT().datasetExplorer;
  const lab = useT().datasetLab;
  const canPick = useRouteOpen(ROUTE.pickFolder);
  const [sourceDir, setSourceDir] = useState("");
  const [pickNote, setPickNote] = useState<string | null>(null);
  const [previewJob, setPreviewJob] =
    useState<IncorporateJob<CocoImportPreview> | null>(null);
  const [applyJob, setApplyJob] =
    useState<IncorporateJob<CocoImportResult> | null>(null);
  const [choices, setChoices] = useState<Record<string, SplitChoice>>({});
  const [plan, setPlan] = useState<CocoImportPlan | null>(null);
  const [planProblem, setPlanProblem] = useState<string | null>(null);
  const [destName, setDestName] = useState("");
  // What the server said about one name; it only counts while that name is typed.
  const [destCheck, setDestCheck] = useState<{
    name: string;
    problem: string | null;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);

  const preview = previewJob?.state === "done" ? previewJob.result : null;
  const previewRunning =
    previewJob !== null && !isJobFinished(previewJob.state);
  const applyRunning = applyJob !== null && !isJobFinished(applyJob.state);
  const result = applyJob?.state === "done" ? applyJob.result : null;
  const previewId = previewJob?.job_id ?? null;

  const fail = useCallback((m: string) => {
    setError(m);
    setCancelling(false);
  }, []);
  const onPreview = useCallback((j: IncorporateJob<CocoImportPreview>) => {
    setPreviewJob(j);
    if (isJobFinished(j.state)) setCancelling(false);
    if (j.state === "done" && j.result) {
      setChoices(initialChoices(j.result.sources));
      setDestName((n) => n || j.result!.suggested_name);
    }
  }, []);
  const onApply = useCallback((j: IncorporateJob<CocoImportResult>) => {
    setApplyJob(j);
    if (isJobFinished(j.state)) setCancelling(false);
  }, []);
  useJobPolling(previewJob, pollPreview, onPreview, fail);
  useJobPolling(applyJob, pollApply, onApply, fail);

  const body = splitsBody(choices);
  const bodyKey = JSON.stringify(body);
  const destOk = isValidDestName(destName);
  const conflicts = splitConflicts(choices);

  // Re-plan whenever a choice changes; the preview itself is not recomputed.
  useEffect(() => {
    if (!previewId || !preview) return;
    let stale = false;
    const timer = setTimeout(() => {
      cocoImportPlan(previewId, body)
        .then((next) => {
          if (stale) return;
          setPlan(next);
          setPlanProblem(null);
        })
        .catch((e: unknown) => {
          if (stale) return;
          setPlan(null);
          setPlanProblem(message(e));
        });
    }, 250);
    return () => {
      stale = true;
      clearTimeout(timer);
    };
    // `body` is covered by `bodyKey`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewId, preview, bodyKey]);

  const destProblem =
    destCheck?.name === destName.trim() ? destCheck.problem : null;

  // The server tells whether the folder name is free; asked a moment after typing.
  useEffect(() => {
    if (!previewId || !preview || !destOk || planProblem !== null) return;
    let stale = false;
    const name = destName.trim();
    const timer = setTimeout(() => {
      cocoImportPlan(previewId, body, name)
        .then(() => {
          if (!stale) setDestCheck({ name, problem: null });
        })
        .catch((err: unknown) => {
          if (!stale) setDestCheck({ name, problem: message(err) });
        });
    }, 300);
    return () => {
      stale = true;
      clearTimeout(timer);
    };
    // `body` is covered by `bodyKey`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewId, preview, destName, destOk, planProblem, bodyKey]);

  const startPreview = async () => {
    setError(null);
    setPreviewJob(null);
    setApplyJob(null);
    setPlan(null);
    setPlanProblem(null);
    setChoices({});
    setDestName("");
    setDestCheck(null);
    try {
      setPreviewJob(await cocoImportPreview(sourceDir));
    } catch (e) {
      setError(message(e));
    }
  };

  const startApply = async () => {
    if (!previewId) return;
    setError(null);
    try {
      setApplyJob(await cocoImportApply(previewId, body, destName));
    } catch (e) {
      setError(message(e));
    }
  };

  const cancel = async (jobId: string | undefined) => {
    if (!jobId) return;
    setCancelling(true);
    try {
      await cocoImportCancel(jobId);
    } catch (e) {
      fail(message(e));
    }
  };

  const working = previewRunning || applyRunning;
  const active = previewRunning ? previewJob : applyJob;
  const canApply =
    canApplyImport(plan, destOk, working || result !== null) &&
    destProblem === null;
  const pending = undecided(choices);

  return (
    <LabTool title={t.importOpen} lead={lab.leadCocoImport} more={t.importIntro}>
      <LabStep title={t.importTitle}>
        <ActionRow
          actions={
            <>
              <button
                type="button"
                hidden={!canPick}
                style={ghostButton}
                title={t.browseTitle}
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
              <AnalyzeButton
                label={t.analyze}
                disabled={working || sourceDir.trim() === ""}
                onClick={() => void startPreview()}
              />
            </>
          }
        >
          <TextField
            label={t.folder}
            value={sourceDir}
            onChange={setSourceDir}
            placeholder={t.folderPlaceholder}
            mono
          />
        </ActionRow>
        <ServerFoldersNote />
        {pickNote && <div style={text}>{pickNote}</div>}
      </LabStep>

      {working && (
        <CocoJobBar
          text={
            previewRunning
              ? t.analyzingImport(previewJob?.done ?? 0, previewJob?.total ?? 0)
              : t.applying(applyJob?.done ?? 0, applyJob?.total ?? 0)
          }
          pct={jobPercent(active?.done ?? 0, active?.total ?? 0)}
          cancelling={cancelling}
          onCancel={() => void cancel(active?.job_id)}
        />
      )}
      {(previewJob?.state === "cancelled" ||
        applyJob?.state === "cancelled") && (
        <div style={text}>{t.cancelled}</div>
      )}
      {previewJob?.state === "failed" && (
        <div style={danger}>{t.failed(previewJob.error ?? "")}</div>
      )}
      {applyJob?.state === "failed" && (
        <div style={danger}>{t.failed(applyJob.error ?? "")}</div>
      )}
      {error && <div style={danger}>{t.failed(error)}</div>}

      {preview && (
        <>
          <LabStep title={t.previewTitle}>
            <div className="vf-lab-cards">
              {preview.sources.map((s) => (
                <SourceCard
                  key={s.name}
                  source={s}
                  choice={choices[s.name] ?? ""}
                  onChoice={(c) => setChoices((prev) => ({ ...prev, [s.name]: c }))}
                />
              ))}
            </div>
            {preview.ignored.length > 0 && (
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <div style={sectionLabel}>{t.ignoredTitle}</div>
                {preview.ignored.map((i) => (
                  <div key={i.name} style={{ ...mono, wordBreak: "break-all" }}>
                    {i.name} — {t.ignoredReasons[i.reason] ?? i.reason}
                  </div>
                ))}
              </div>
            )}
            {conflicts.map((s) => (
              <div key={s} style={danger}>
                {t.splitConflict(ex.splitNames[s])}
              </div>
            ))}
            {pending > 0 && <div style={warn}>{t.unassigned(pending)}</div>}
            {planProblem && <div style={danger}>{t.failed(planProblem)}</div>}
          </LabStep>
          {plan && (
            <LabStep title={t.planTitle}>
              <PlanView plan={plan} />
            </LabStep>
          )}
          <DestinationRow
            name={destName}
            onName={setDestName}
            ok={destOk}
            canApply={canApply}
            applyLabel={t.applyImport}
            problem={destProblem}
            onApply={() => void startApply()}
          />
        </>
      )}

      {result && (
        <div className="vf-lab-done">
          <h3 style={{ ...stepTitle, margin: 0 }}>{t.doneTitle}</h3>
          <div style={mono}>
            {t.doneImport(
              result.images,
              result.boxes,
              result.classes,
              result.path,
            )}
          </div>
          <div style={mono}>
            {t.linkedCopied(result.files_linked, result.files_copied)}
          </div>
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
