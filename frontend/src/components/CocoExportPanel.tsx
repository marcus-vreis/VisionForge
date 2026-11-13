import { useCallback, useEffect, useState } from "react";
import {
  cocoExportApply,
  cocoExportCancel,
  cocoExportPlan,
  cocoExportPreview,
  cocoExportStatus,
} from "../api/client";
import { SPLITS, type ExplorerForm } from "../lib/dataset-explorer";
import {
  isJobFinished,
  isValidDestName,
  jobPercent,
  type IncorporateJob,
} from "../lib/dataset-incorporate";
import type { CocoExportPreview, CocoExportResult } from "../lib/dataset-coco";
import { errorMessage as message, useJobPolling } from "../lib/useJobPolling";
import { useT } from "../i18n/useT";
import { CocoJobBar, DestinationRow, ReasonSummary } from "./CocoParts";
import { danger, mono, sectionLabel, stepTitle, text } from "./datasetExplorerStyles";
import { ActionRow, AnalyzeButton, DataTable, LabStep, LabTool } from "./DatasetLabParts";

const pollPreview = (id: string) => cocoExportStatus<CocoExportPreview>(id);
const pollApply = (id: string) => cocoExportStatus<CocoExportResult>(id);

/** "Exportar para COCO" (ADR-169): write the opened YOLO dataset as a new COCO
 *  dataset next to it. The explorer does not read COCO, so the result is only
 *  reported, not opened. */
export function CocoExportPanel({ form }: { form: ExplorerForm }) {
  const t = useT().datasetCoco;
  const ex = useT().datasetExplorer;
  const lab = useT().datasetLab;
  const [previewJob, setPreviewJob] =
    useState<IncorporateJob<CocoExportPreview> | null>(null);
  const [applyJob, setApplyJob] =
    useState<IncorporateJob<CocoExportResult> | null>(null);
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
  const onPreview = useCallback((j: IncorporateJob<CocoExportPreview>) => {
    setPreviewJob(j);
    if (isJobFinished(j.state)) setCancelling(false);
    if (j.state === "done" && j.result)
      setDestName((n) => n || j.result!.suggested_name);
  }, []);
  const onApply = useCallback((j: IncorporateJob<CocoExportResult>) => {
    setApplyJob(j);
    if (isJobFinished(j.state)) setCancelling(false);
  }, []);
  useJobPolling(previewJob, pollPreview, onPreview, fail);
  useJobPolling(applyJob, pollApply, onApply, fail);

  const destOk = isValidDestName(destName);

  const destProblem =
    destCheck?.name === destName.trim() ? destCheck.problem : null;

  // The server tells whether the folder name is free; asked a moment after typing.
  useEffect(() => {
    if (!previewId || !preview || !destOk) return;
    let stale = false;
    const name = destName.trim();
    const timer = setTimeout(() => {
      cocoExportPlan(previewId, name)
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
  }, [previewId, preview, destName, destOk]);

  const startPreview = async () => {
    setError(null);
    setPreviewJob(null);
    setApplyJob(null);
    setDestName("");
    setDestCheck(null);
    try {
      setPreviewJob(await cocoExportPreview(form));
    } catch (e) {
      setError(message(e));
    }
  };

  const startApply = async () => {
    if (!previewId) return;
    setError(null);
    try {
      setApplyJob(await cocoExportApply(previewId, destName));
    } catch (e) {
      setError(message(e));
    }
  };

  const cancel = async (jobId: string | undefined) => {
    if (!jobId) return;
    setCancelling(true);
    try {
      await cocoExportCancel(jobId);
    } catch (e) {
      fail(message(e));
    }
  };

  const canApply =
    preview !== null &&
    destOk &&
    destProblem === null &&
    !applyRunning &&
    !result;
  const working = previewRunning || applyRunning;
  const active = previewRunning ? previewJob : applyJob;

  return (
    <LabTool title={t.exportOpen} lead={lab.leadCocoExport} more={t.exportIntro}>
      <LabStep title={t.exportTitle}>
        <ActionRow
          actions={
            <AnalyzeButton
              label={t.analyze}
              disabled={working}
              onClick={() => void startPreview()}
            />
          }
        />
      </LabStep>

      {working && (
        <CocoJobBar
          text={
            previewRunning
              ? t.analyzingExport(previewJob?.done ?? 0, previewJob?.total ?? 0)
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
            <div style={{ display: "flex", gap: 18, flexWrap: "wrap", ...mono }}>
              <span>{t.imagesLine(preview.images)}</span>
              <span>{t.boxesLine(preview.boxes)}</span>
              <span>{t.categoriesLine(preview.categories)}</span>
            </div>
            <DataTable maxWidth={420}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left" }}>{t.splitColumn}</th>
                  <th style={{ textAlign: "right" }}>{t.imagesColumn}</th>
                  <th style={{ textAlign: "right" }}>{t.boxesColumn}</th>
                </tr>
              </thead>
              <tbody>
                {SPLITS.filter((s) => preview.present[s]).map((s) => (
                  <tr key={s}>
                    <td>{ex.splitNames[s]}</td>
                    <td style={{ textAlign: "right" }}>
                      {preview.per_split[s].images}
                    </td>
                    <td style={{ textAlign: "right" }}>
                      {preview.per_split[s].boxes}
                    </td>
                  </tr>
                ))}
              </tbody>
            </DataTable>
            {preview.excluded > 0 && (
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <div style={sectionLabel}>
                  {t.excludedTitle(preview.excluded)}
                </div>
                <div style={text}>{t.excludedHelp}</div>
                <ReasonSummary
                  counts={preview.excluded_by_reason}
                  labels={t.exportReasons}
                />
                {preview.excluded_listed.map((e) => (
                  <div key={e.path} style={{ ...mono, wordBreak: "break-all" }}>
                    {e.path} — {t.exportReasons[e.reason] ?? e.reason}
                    {e.detail ? ` (${e.detail})` : ""}
                  </div>
                ))}
                {preview.excluded > preview.excluded_listed.length && (
                  <div style={mono}>
                    {t.moreNotShown(
                      preview.excluded - preview.excluded_listed.length,
                    )}
                  </div>
                )}
              </div>
            )}
          </LabStep>
          <DestinationRow
            name={destName}
            onName={setDestName}
            ok={destOk}
            canApply={canApply}
            applyLabel={t.applyExport}
            problem={destProblem}
            onApply={() => void startApply()}
          />
        </>
      )}

      {result && (
        <div className="vf-lab-done">
          <h3 style={{ ...stepTitle, margin: 0 }}>{t.doneTitle}</h3>
          <div style={mono}>
            {t.doneExport(
              result.images,
              result.boxes,
              result.excluded,
              result.path,
            )}
          </div>
          <div style={mono}>
            {t.linkedCopied(result.files_linked, result.files_copied)}
          </div>
        </div>
      )}
    </LabTool>
  );
}
