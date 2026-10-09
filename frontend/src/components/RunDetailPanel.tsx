import { useEffect, useState, type CSSProperties, type ReactNode } from "react";
import {
  ApiError,
  artifactUrl,
  batchPredictRun,
  downloadRunMarkdown,
  exportRunToOnnx,
  fetchRunDetail,
  resumeRun,
  revealRunFolder,
  gradcamRun,
  pickDatasetFolder,
  testRunOnDataset,
  type BatchPredictResponse,
  type ExportOnnxResponse,
  type GradCamItem,
  type GradCamResponse,
  type RunDetail,
  type TestRecord,
} from "../api/client";
import type { Dict } from "../i18n/pt";
import { useI18n, useT } from "../i18n/useT";
import { formatBytes, shortDigest } from "../lib/dataset-identity";
import { metricCi } from "../lib/metric-ci";
import { pickerCancelText } from "../lib/picker-feedback";
import { PREPROCESS_KIND_LABELS } from "../lib/preprocess-kinds";
import { canOfferReveal, revealErrorText } from "../lib/reveal-folder";
import { stdDdof } from "../lib/cv-std";
import { STOPPED_COLOR, unitState } from "../lib/unit-status";
import type { MetricCI } from "../types/run";
import { Lightbox } from "./Lightbox";

interface RunDetailPanelProps {
  runId: string;
  onBack: () => void;
}

/** Metric names read the same in every language; only the words around them are translated. */
const METRIC_NAMES: Record<string, string> = {
  f1: "F1",
  recall: "Recall",
  auc_roc: "AUC-ROC",
  // Detection metrics (mAP @ IoU thresholds; box validation loss).
  map50: "mAP@50",
  map50_95: "mAP@50-95",
  box_loss: "Box loss (val)",
};

/** The metrics a test run reports under a `test_` prefix. */
const TEST_METRICS = ["accuracy", "f1", "precision", "recall", "auc_roc"];

function metricLabel(t: Dict, key: string): string {
  const words: Record<string, string> = t.runDetail.metrics.labels;
  const label = METRIC_NAMES[key] ?? words[key];
  if (label) return label;
  if (key.startsWith("test_") && TEST_METRICS.includes(key.slice(5))) {
    return t.runDetail.metrics.onTestSet(metricLabel(t, key.slice(5)));
  }
  return key;
}

function fmtMetric(v: unknown): string {
  if (v === null || v === undefined) return "—";
  // A metric that was never measured is null; a NaN or an infinity (the old
  // sentinel of a run with no best epoch) must not reach the screen either.
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "—";
    return v % 1 === 0 ? String(v) : v.toFixed(4);
  }
  return String(v);
}

interface ConfigData {
  preprocessing?: { steps?: Array<Record<string, unknown>> };
  transforms?: Record<string, unknown>;
}

function getDataSection(config: Record<string, unknown>): ConfigData | null {
  const data = config["data"];
  if (data === null || typeof data !== "object" || Array.isArray(data)) return null;
  return data as ConfigData;
}

/** Return a nested object section of the run config, or null when absent/non-object. */
function getConfigRecord(
  config: Record<string, unknown>,
  key: string,
): Record<string, unknown> | null {
  const value = config[key];
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

export function RunDetailPanel({ runId, onBack }: RunDetailPanelProps) {
  const { t, locale } = useI18n();
  const graphLabels: Record<string, string> = t.plots.labels;
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lightbox, setLightbox] = useState<{ src: string; caption: string } | null>(null);
  const [testForm, setTestForm] = useState({
    data_dir: "",
    label: "",
  });
  const [testing, setTesting] = useState(false);
  const [testMsg, setTestMsg] = useState<{ kind: "info" | "error" | "success"; text: string } | null>(null);
  const [showTestForm, setShowTestForm] = useState(false);

  // ONNX export state — kept local because the result is single-shot, not part
  // of the persistent run.json.
  const [showExportForm, setShowExportForm] = useState(false);
  const [exportForm, setExportForm] = useState({
    opset_version: 17,
    dynamic_axes: true,
    validate: true,
    benchmark: true,
    benchmark_runs: 50,
  });
  const [exporting, setExporting] = useState(false);
  const [exportResult, setExportResult] = useState<ExportOnnxResponse | null>(null);
  const [exportMsg, setExportMsg] = useState<{ kind: "info" | "error" | "success"; text: string } | null>(null);

  // Batch prediction state — independent of test/export so the user can run
  // each separately and review results.
  const [showBatchForm, setShowBatchForm] = useState(false);
  const [batchForm, setBatchForm] = useState({
    input_dir: "",
    recursive: true,
  });
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchResult, setBatchResult] = useState<BatchPredictResponse | null>(null);
  const [batchMsg, setBatchMsg] = useState<{ kind: "info" | "error" | "success"; text: string } | null>(null);

  // Resume state — a run that stopped short can be continued in its own
  // directory (ADR-092/093); the config comes from its run.json, not from here.
  const [resuming, setResuming] = useState(false);
  const [resumeMsg, setResumeMsg] = useState<
    { kind: "info" | "error" | "success"; text: string } | null
  >(null);

  // Open-folder state — the server opens the file manager on its own desktop,
  // so the button only exists when the server said that desktop is the user's.
  const [revealing, setRevealing] = useState(false);
  const [revealMsg, setRevealMsg] = useState<
    { kind: "info" | "error" | "success"; text: string } | null
  >(null);

  // Grad-CAM state — independent single-shot explainability action.
  const [showGradcamForm, setShowGradcamForm] = useState(false);
  const [gradcamForm, setGradcamForm] = useState({ input_dir: "", num_samples: 8 });
  const [gradcamRunning, setGradcamRunning] = useState(false);
  const [gradcamResult, setGradcamResult] = useState<GradCamResponse | null>(null);
  const [gradcamMsg, setGradcamMsg] = useState<{ kind: "info" | "error" | "success"; text: string } | null>(null);

  // A detection run.json carries task="detection". Some post-training actions
  // (batch CSV inference, per-model evaluate) are classification-only and hidden
  // for detection runs.
  const task = runTask(detail);
  const isDetection = task === "detection";
  // A custom task owns its own training loop, so the built-in post-training
  // actions have no checkpoint contract to load; the API rejects them with 400.
  const isCustom = task.startsWith("custom:");
  const detectionBackend = (
    detail?.config?.["model"] as Record<string, unknown> | undefined
  )?.["backend"] as string | undefined;
  // ONNX export: classification + regression + segmentation (shared core helper)
  // and Ultralytics detection (torchvision detection export is not implemented).
  // Anomaly is excluded — PatchCore's memory-bank scoring has no forward graph.
  const canExportOnnx = isDetection
    ? detectionBackend === "ultralytics"
    : !isCustom && task !== "anomaly";
  // Batch CSV inference: classification + regression (continuous targets) +
  // anomaly (score + threshold decision). Detection/segmentation produce
  // per-box/per-pixel outputs that don't map to a flat row yet (ADR-041 slice 3).
  const canBatchPredict = !isDetection && !isCustom && task !== "segmentation";
  // Grad-CAM: classification + regression + segmentation (conv-based CAM, ADR-053).
  // Detection (Ultralytics) and anomaly (no class/conv target) are excluded.
  const canGradcam = !isDetection && !isCustom && task !== "anomaly";

  useEffect(() => {
    let alive = true;
    setLoading(true);
    fetchRunDetail(runId)
      .then((d) => {
        if (alive) setDetail(d);
      })
      .catch((e: unknown) => {
        if (!alive) return;
        // Empty means "no message from the server": the banner then shows its own text.
        setError(e instanceof Error ? e.message : "");
      })
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [runId]);

  const doResume = async () => {
    setResuming(true);
    setResumeMsg({ kind: "info", text: t.runDetail.resume.queuing });
    try {
      const res = await resumeRun(runId);
      setResumeMsg({
        kind: "success",
        text:
          res.status === "running"
            ? t.runDetail.resume.running
            : t.runDetail.resume.queued,
      });
      await reload();
    } catch (e) {
      setResumeMsg({
        kind: "error",
        text: e instanceof Error ? e.message : t.runDetail.resume.failed,
      });
    } finally {
      setResuming(false);
    }
  };

  const doReveal = async () => {
    setRevealing(true);
    setRevealMsg(null);
    try {
      await revealRunFolder(runId);
      setRevealMsg({ kind: "success", text: t.runDetail.reveal.opened });
    } catch (e) {
      setRevealMsg({
        kind: "error",
        text: revealErrorText(
          e instanceof ApiError ? e.status : 0,
          t.runDetail.reveal,
        ),
      });
    } finally {
      setRevealing(false);
    }
  };

  const reload = async () => {
    try {
      const d = await fetchRunDetail(runId);
      setDetail(d);
    } catch {
      /* ignore — keep existing detail */
    }
  };

  const pickFolder = async () => {
    setTestMsg({ kind: "info", text: t.runDetail.picker.opening });
    try {
      const res = await pickDatasetFolder();
      if (res.cancelled) {
        setTestMsg({ kind: "info", text: pickerCancelText(res, t.runDetail.picker.cancelled) });
        return;
      }
      setTestForm((f) => ({ ...f, data_dir: res.path }));
      setTestMsg({ kind: "success", text: t.runDetail.picker.picked(res.path) });
    } catch (e) {
      const msg = e instanceof Error ? e.message : t.runDetail.picker.failed;
      setTestMsg({ kind: "error", text: msg });
    }
  };

  const pickBatchFolder = async () => {
    setBatchMsg({ kind: "info", text: t.runDetail.picker.opening });
    try {
      const res = await pickDatasetFolder();
      if (res.cancelled) {
        setBatchMsg({ kind: "info", text: pickerCancelText(res, t.runDetail.picker.cancelled) });
        return;
      }
      setBatchForm((f) => ({ ...f, input_dir: res.path }));
      setBatchMsg({ kind: "success", text: t.runDetail.picker.picked(res.path) });
    } catch (e) {
      setBatchMsg({
        kind: "error",
        text: e instanceof Error ? e.message : t.runDetail.picker.failed,
      });
    }
  };

  const runBatch = async () => {
    if (!batchForm.input_dir.trim()) {
      setBatchMsg({
        kind: "error",
        text: t.runDetail.batch.needFolder,
      });
      return;
    }
    setBatchRunning(true);
    setBatchResult(null);
    setBatchMsg({ kind: "info", text: t.runDetail.batch.starting });
    try {
      const result = await batchPredictRun(runId, {
        input_dir: batchForm.input_dir,
        recursive: batchForm.recursive,
      });
      setBatchResult(result);
      const okCount = result.total_processed;
      const failed = result.failed_count;
      setBatchMsg({
        kind: failed === 0 ? "success" : "info",
        text:
          failed === 0
            ? t.runDetail.batch.done(okCount, result.output_csv)
            : t.runDetail.batch.doneWithFailures(okCount, failed, result.output_csv),
      });
    } catch (e) {
      const msg =
        e instanceof ApiError
          ? e.message
          : e instanceof Error
            ? e.message
            : t.runDetail.batch.failed;
      setBatchMsg({ kind: "error", text: msg });
    } finally {
      setBatchRunning(false);
    }
  };

  const pickGradcamFolder = async () => {
    setGradcamMsg({ kind: "info", text: t.runDetail.picker.opening });
    try {
      const res = await pickDatasetFolder();
      if (res.cancelled) {
        setGradcamMsg({ kind: "info", text: pickerCancelText(res, t.runDetail.picker.cancelled) });
        return;
      }
      setGradcamForm((f) => ({ ...f, input_dir: res.path }));
      setGradcamMsg({ kind: "success", text: t.runDetail.picker.picked(res.path) });
    } catch (e) {
      setGradcamMsg({
        kind: "error",
        text: e instanceof Error ? e.message : t.runDetail.picker.failed,
      });
    }
  };

  const runGradcam = async () => {
    if (!gradcamForm.input_dir.trim()) {
      setGradcamMsg({ kind: "error", text: t.runDetail.gradcam.needFolder });
      return;
    }
    setGradcamRunning(true);
    setGradcamResult(null);
    setGradcamMsg({ kind: "info", text: t.runDetail.gradcam.starting });
    try {
      const result = await gradcamRun(runId, {
        input_dir: gradcamForm.input_dir,
        num_samples: gradcamForm.num_samples,
      });
      setGradcamResult(result);
      setGradcamMsg({
        kind: "success",
        text: t.runDetail.gradcam.done(result.count, result.target_layer),
      });
    } catch (e) {
      const msg =
        e instanceof ApiError
          ? e.message
          : e instanceof Error
            ? e.message
            : t.runDetail.gradcam.failed;
      setGradcamMsg({ kind: "error", text: msg });
    } finally {
      setGradcamRunning(false);
    }
  };

  const runExport = async () => {
    setExporting(true);
    setExportMsg({ kind: "info", text: t.runDetail.onnx.exporting });
    setExportResult(null);
    try {
      const result = await exportRunToOnnx(runId, {
        opset_version: exportForm.opset_version,
        dynamic_axes: exportForm.dynamic_axes,
        validate: exportForm.validate,
        benchmark: exportForm.benchmark,
        benchmark_runs: exportForm.benchmark_runs,
      });
      setExportResult(result);
      setExportMsg({
        kind: "success",
        text: t.runDetail.onnx.saved(result.output_onnx),
      });
    } catch (e) {
      const msg =
        e instanceof ApiError
          ? e.message
          : e instanceof Error
            ? e.message
            : t.runDetail.onnx.failed;
      setExportMsg({ kind: "error", text: msg });
    } finally {
      setExporting(false);
    }
  };

  const runTest = async () => {
    if (!testForm.data_dir.trim()) {
      setTestMsg({ kind: "error", text: t.runDetail.tests.needFolder });
      return;
    }
    setTesting(true);
    setTestMsg({ kind: "info", text: t.runDetail.tests.starting });
    try {
      const record = await testRunOnDataset(runId, {
        data_dir: testForm.data_dir,
        label: testForm.label || undefined,
      });
      setTestMsg({
        kind: "success",
        text: t.runDetail.tests.recorded(record.test_id),
      });
      setShowTestForm(false);
      await reload();
    } catch (e) {
      const msg =
        e instanceof ApiError ? e.message : e instanceof Error ? e.message : t.runDetail.tests.failed;
      setTestMsg({ kind: "error", text: msg });
    } finally {
      setTesting(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <button
          type="button"
          onClick={onBack}
          style={{
            padding: "6px 12px",
            background: "rgba(255,255,255,0.04)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 8,
            color: "var(--vf-text-dim)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            letterSpacing: "0.10em",
            textTransform: "uppercase",
            cursor: "pointer",
          }}
        >
          {t.runDetail.back}
        </button>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: 14, color: "var(--vf-text)" }}>
          {runId}
        </div>
        {detail?.resumable && (
          <button
            type="button"
            onClick={() => void doResume()}
            disabled={resuming}
            title={
              detail.configured_epochs
                ? t.runDetail.resume.title(
                    (detail.metrics?.["total_epochs"] as number) ?? 0,
                    detail.configured_epochs,
                  )
                : t.runDetail.resume.titleNoTotal
            }
            style={{
              marginLeft: "auto",
              padding: "6px 12px",
              background: "oklch(0.80 0.16 85 / 0.14)",
              border: "1px solid oklch(0.80 0.16 85 / 0.45)",
              borderRadius: 8,
              color: "oklch(0.90 0.15 85)",
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              letterSpacing: "0.10em",
              textTransform: "uppercase",
              cursor: resuming ? "wait" : "pointer",
              opacity: resuming ? 0.6 : 1,
            }}
          >
            {t.runDetail.resume.button}
            {detail.configured_epochs
              ? ` ${(detail.metrics?.["total_epochs"] as number) ?? 0}/${detail.configured_epochs}`
              : ""}
          </button>
        )}
        <button
          type="button"
          onClick={() => void downloadRunMarkdown(runId)}
          title={t.modelCard.title}
          style={{
            marginLeft: detail?.resumable ? 0 : "auto",
            padding: "6px 12px",
            background: "var(--accent-soft)",
            border: "1px solid var(--accent-vf)",
            borderRadius: 8,
            color: "var(--vf-text)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            letterSpacing: "0.10em",
            textTransform: "uppercase",
            cursor: "pointer",
          }}
        >
          {t.modelCard.button}
        </button>
      </div>

      {resumeMsg && (
        <div
          style={{
            padding: "8px 12px",
            borderRadius: 8,
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            background:
              resumeMsg.kind === "error"
                ? "oklch(0.704 0.191 22.216 / 0.10)"
                : "rgba(255,255,255,0.04)",
            border: "1px solid var(--vf-panel-stroke)",
            color:
              resumeMsg.kind === "error"
                ? "oklch(0.80 0.17 22)"
                : "var(--vf-text-dim)",
          }}
        >
          {resumeMsg.text}
        </div>
      )}

      {loading && (
        <div style={{ padding: 32, textAlign: "center", color: "var(--vf-text-muted)" }}>
          {t.runDetail.loading}
        </div>
      )}

      {error !== null && (
        <div
          style={{
            padding: 14,
            background: "oklch(0.704 0.191 22.216 / 0.10)",
            border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
            borderRadius: 10,
            color: "oklch(0.85 0.14 22)",
            fontFamily: "var(--font-mono)",
            fontSize: 12,
          }}
        >
          {error || t.runDetail.loadFailed}
        </div>
      )}

      {detail && (
        <>
          {detail.dataset && (
            <Section title={t.runDetail.dataset.title}>
              <KeyRow label={t.runDetail.dataset.name} value={detail.dataset.name} />
              <PathRow label={t.runDetail.dataset.path} value={detail.dataset.root} />
              {detail.dataset.digest ? (
                <>
                  <KeyRow
                    label={t.runDetail.dataset.contents}
                    value={t.runDetail.dataset.files(
                      detail.dataset.n_files,
                      formatBytes(detail.dataset.total_bytes, locale),
                    )}
                  />
                  <KeyRow
                    label={t.runDetail.dataset.fingerprint}
                    value={`${detail.dataset.method} ${shortDigest(detail.dataset.digest)}`}
                  />
                </>
              ) : (
                // Saying nothing here would read as "no dataset"; the run has one,
                // it just predates the fingerprint (ADR-061, 2026-07-26).
                <KeyRow
                  label={t.runDetail.dataset.fingerprint}
                  value={t.runDetail.dataset.noFingerprint}
                />
              )}
            </Section>
          )}

          <Section title={t.runDetail.location.title}>
            <PathRow
              label={t.runDetail.location.runFolder}
              value={detail.run_dir}
              action={
                canOfferReveal(detail) ? (
                  <button
                    type="button"
                    onClick={() => void doReveal()}
                    disabled={revealing}
                    title={t.runDetail.reveal.title}
                    style={{ ...PATH_ROW_BUTTON_STYLE, opacity: revealing ? 0.6 : 1 }}
                  >
                    {t.runDetail.reveal.button}
                  </button>
                ) : null
              }
            />
            {revealMsg && (
              <div
                role={revealMsg.kind === "error" ? "alert" : "status"}
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  padding: "4px 0",
                  color:
                    revealMsg.kind === "error"
                      ? "oklch(0.80 0.17 22)"
                      : "var(--vf-text-dim)",
                }}
              >
                {revealMsg.text}
              </div>
            )}
            {detail.artifacts.model && (
              <PathRow label={t.runDetail.location.checkpoint} value={detail.artifacts.model} />
            )}
            {detail.device_used && (
              <KeyRow label={t.runDetail.location.deviceUsed} value={detail.device_used} />
            )}
            {detail.environment &&
              Object.entries(detail.environment).map(([k, v]) => (
                <KeyRow key={k} label={t.runDetail.location.env(k)} value={v} />
              ))}
          </Section>

          <TrainingSection config={detail.config} />

          <PipelineSection config={detail.config} />

          {detail.artifacts.model && canExportOnnx && (
            <Section
              title={t.runDetail.onnx.title}
              action={
                <button
                  type="button"
                  onClick={() => {
                    setShowExportForm((s) => !s);
                    setExportMsg(null);
                  }}
                  style={{
                    padding: "6px 12px",
                    background: "var(--accent-soft)",
                    border: "1px solid var(--accent-vf)",
                    borderRadius: 8,
                    color: "var(--vf-text)",
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    cursor: "pointer",
                    letterSpacing: "0.10em",
                    textTransform: "uppercase",
                  }}
                >
                  {showExportForm ? t.common.cancel : t.runDetail.onnx.open}
                </button>
              }
            >
              {showExportForm ? (
                <div
                  style={{
                    padding: 14,
                    border: "1px dashed var(--vf-panel-stroke)",
                    borderRadius: 10,
                    display: "flex",
                    flexDirection: "column",
                    gap: 10,
                  }}
                >
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(2, 1fr)",
                      gap: 10,
                    }}
                  >
                    <label style={exportLabelStyle}>
                      <span>{t.runDetail.onnx.opsetVersion}</span>
                      <input
                        type="number"
                        min={11}
                        max={20}
                        value={exportForm.opset_version}
                        onChange={(e) =>
                          setExportForm((f) => ({
                            ...f,
                            opset_version: parseInt(e.target.value, 10) || 17,
                          }))
                        }
                        style={exportInputStyle}
                      />
                    </label>
                    <label style={exportLabelStyle}>
                      <span>{t.runDetail.onnx.benchmarkRuns}</span>
                      <input
                        type="number"
                        min={5}
                        value={exportForm.benchmark_runs}
                        onChange={(e) =>
                          setExportForm((f) => ({
                            ...f,
                            benchmark_runs: parseInt(e.target.value, 10) || 50,
                          }))
                        }
                        disabled={!exportForm.benchmark}
                        style={{
                          ...exportInputStyle,
                          opacity: exportForm.benchmark ? 1 : 0.4,
                        }}
                      />
                    </label>
                  </div>
                  <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
                    <ExportToggle
                      label={t.runDetail.onnx.dynamicAxes}
                      value={exportForm.dynamic_axes}
                      onChange={(v) =>
                        setExportForm((f) => ({ ...f, dynamic_axes: v }))
                      }
                    />
                    <ExportToggle
                      label={t.runDetail.onnx.validate}
                      value={exportForm.validate}
                      onChange={(v) =>
                        setExportForm((f) => ({ ...f, validate: v }))
                      }
                    />
                    <ExportToggle
                      label={t.runDetail.onnx.benchmark}
                      value={exportForm.benchmark}
                      onChange={(v) =>
                        setExportForm((f) => ({ ...f, benchmark: v }))
                      }
                    />
                  </div>
                  <button
                    type="button"
                    onClick={() => void runExport()}
                    disabled={exporting}
                    style={{
                      padding: "12px 20px",
                      background:
                        "linear-gradient(180deg, var(--accent-soft) 0%, rgba(8,10,14,0.4) 100%)",
                      border: "1px solid var(--accent-vf)",
                      borderRadius: 10,
                      color: "var(--vf-text)",
                      fontFamily: "var(--font-mono)",
                      fontSize: 12,
                      fontWeight: 600,
                      letterSpacing: "0.10em",
                      textTransform: "uppercase",
                      cursor: exporting ? "wait" : "pointer",
                      opacity: exporting ? 0.6 : 1,
                    }}
                  >
                    {exporting ? t.runDetail.onnx.running : t.runDetail.onnx.run}
                  </button>
                  {exportMsg && (
                    <div
                      style={{
                        padding: "8px 12px",
                        fontFamily: "var(--font-mono)",
                        fontSize: 11,
                        borderRadius: 8,
                        background:
                          exportMsg.kind === "error"
                            ? "oklch(0.704 0.191 22.216 / 0.10)"
                            : exportMsg.kind === "success"
                              ? "oklch(0.72 0.16 150 / 0.10)"
                              : "rgba(255,255,255,0.04)",
                        color:
                          exportMsg.kind === "error"
                            ? "oklch(0.85 0.14 22)"
                            : exportMsg.kind === "success"
                              ? "oklch(0.85 0.16 150)"
                              : "var(--vf-text-dim)",
                        wordBreak: "break-all",
                      }}
                    >
                      {exportMsg.text}
                    </div>
                  )}
                </div>
              ) : (
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    color: "var(--vf-text-muted)",
                    lineHeight: 1.5,
                  }}
                >
                  {t.runDetail.onnx.hint} <code>{t.runDetail.onnx.file}</code>.
                </div>
              )}
              {exportResult && (
                <ExportResultPanel result={exportResult} />
              )}
            </Section>
          )}

          {detail.artifacts.model && canBatchPredict && (
            <Section
              title={t.runDetail.batch.title}
              action={
                <button
                  type="button"
                  onClick={() => {
                    setShowBatchForm((s) => !s);
                    setBatchMsg(null);
                  }}
                  style={{
                    padding: "6px 12px",
                    background: "var(--accent-soft)",
                    border: "1px solid var(--accent-vf)",
                    borderRadius: 8,
                    color: "var(--vf-text)",
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    cursor: "pointer",
                    letterSpacing: "0.10em",
                    textTransform: "uppercase",
                  }}
                >
                  {showBatchForm ? t.common.cancel : t.runDetail.batch.open}
                </button>
              }
            >
              {showBatchForm ? (
                <div
                  style={{
                    padding: 14,
                    border: "1px dashed var(--vf-panel-stroke)",
                    borderRadius: 10,
                    display: "flex",
                    flexDirection: "column",
                    gap: 10,
                  }}
                >
                  <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
                    <FormField
                      label={t.runDetail.imageFolder}
                      value={batchForm.input_dir}
                      onChange={(v) =>
                        setBatchForm((f) => ({ ...f, input_dir: v }))
                      }
                      placeholder={t.runDetail.batch.folderPlaceholder}
                    />
                    <button
                      type="button"
                      onClick={() => void pickBatchFolder()}
                      style={{
                        padding: "10px 14px",
                        background: "transparent",
                        border: "1px solid var(--vf-panel-stroke)",
                        borderRadius: 10,
                        color: "var(--vf-text-dim)",
                        fontFamily: "var(--font-mono)",
                        fontSize: 11,
                        cursor: "pointer",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {t.runDetail.browse}
                    </button>
                  </div>
                  <ExportToggle
                    label={t.runDetail.batch.recursive}
                    value={batchForm.recursive}
                    onChange={(v) =>
                      setBatchForm((f) => ({ ...f, recursive: v }))
                    }
                  />
                  <button
                    type="button"
                    onClick={() => void runBatch()}
                    disabled={batchRunning}
                    style={{
                      padding: "12px 20px",
                      background:
                        "linear-gradient(180deg, var(--accent-soft) 0%, rgba(8,10,14,0.4) 100%)",
                      border: "1px solid var(--accent-vf)",
                      borderRadius: 10,
                      color: "var(--vf-text)",
                      fontFamily: "var(--font-mono)",
                      fontSize: 12,
                      fontWeight: 600,
                      letterSpacing: "0.10em",
                      textTransform: "uppercase",
                      cursor: batchRunning ? "wait" : "pointer",
                      opacity: batchRunning ? 0.6 : 1,
                    }}
                  >
                    {batchRunning ? t.runDetail.batch.running : t.runDetail.batch.run}
                  </button>
                  {batchMsg && (
                    <div
                      style={{
                        padding: "8px 12px",
                        fontFamily: "var(--font-mono)",
                        fontSize: 11,
                        borderRadius: 8,
                        background:
                          batchMsg.kind === "error"
                            ? "oklch(0.704 0.191 22.216 / 0.10)"
                            : batchMsg.kind === "success"
                              ? "oklch(0.72 0.16 150 / 0.10)"
                              : "rgba(255,255,255,0.04)",
                        color:
                          batchMsg.kind === "error"
                            ? "oklch(0.85 0.14 22)"
                            : batchMsg.kind === "success"
                              ? "oklch(0.85 0.16 150)"
                              : "var(--vf-text-dim)",
                        wordBreak: "break-all",
                      }}
                    >
                      {batchMsg.text}
                    </div>
                  )}
                </div>
              ) : (
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    color: "var(--vf-text-muted)",
                    lineHeight: 1.5,
                  }}
                >
                  {t.runDetail.batch.hint}
                </div>
              )}
              {batchResult && <BatchResultPanel result={batchResult} />}
            </Section>
          )}

          {detail.artifacts.model && canGradcam && (
            <Section
              title={t.runDetail.gradcam.title}
              action={
                <button
                  type="button"
                  onClick={() => {
                    setShowGradcamForm((s) => !s);
                    setGradcamMsg(null);
                  }}
                  style={{
                    padding: "6px 12px",
                    background: "var(--accent-soft)",
                    border: "1px solid var(--accent-vf)",
                    borderRadius: 8,
                    color: "var(--vf-text)",
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    cursor: "pointer",
                    letterSpacing: "0.10em",
                    textTransform: "uppercase",
                  }}
                >
                  {showGradcamForm ? t.common.cancel : t.runDetail.gradcam.open}
                </button>
              }
            >
              {showGradcamForm ? (
                <div
                  style={{
                    padding: 14,
                    border: "1px dashed var(--vf-panel-stroke)",
                    borderRadius: 10,
                    display: "flex",
                    flexDirection: "column",
                    gap: 10,
                  }}
                >
                  <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
                    <FormField
                      label={t.runDetail.imageFolder}
                      value={gradcamForm.input_dir}
                      onChange={(v) =>
                        setGradcamForm((f) => ({ ...f, input_dir: v }))
                      }
                      placeholder={t.runDetail.gradcam.folderPlaceholder}
                    />
                    <button
                      type="button"
                      onClick={() => void pickGradcamFolder()}
                      style={{
                        padding: "10px 14px",
                        background: "transparent",
                        border: "1px solid var(--vf-panel-stroke)",
                        borderRadius: 10,
                        color: "var(--vf-text-dim)",
                        fontFamily: "var(--font-mono)",
                        fontSize: 11,
                        cursor: "pointer",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {t.runDetail.browse}
                    </button>
                  </div>
                  <FormField
                    label={t.runDetail.gradcam.samples}
                    value={String(gradcamForm.num_samples)}
                    onChange={(v) =>
                      setGradcamForm((f) => ({
                        ...f,
                        num_samples: Math.max(1, Math.min(64, Number(v) || 1)),
                      }))
                    }
                  />
                  <button
                    type="button"
                    onClick={() => void runGradcam()}
                    disabled={gradcamRunning}
                    style={{
                      padding: "12px 20px",
                      background:
                        "linear-gradient(180deg, var(--accent-soft) 0%, rgba(8,10,14,0.4) 100%)",
                      border: "1px solid var(--accent-vf)",
                      borderRadius: 10,
                      color: "var(--vf-text)",
                      fontFamily: "var(--font-mono)",
                      fontSize: 12,
                      fontWeight: 600,
                      letterSpacing: "0.10em",
                      textTransform: "uppercase",
                      cursor: gradcamRunning ? "wait" : "pointer",
                      opacity: gradcamRunning ? 0.6 : 1,
                    }}
                  >
                    {gradcamRunning ? t.runDetail.gradcam.running : t.runDetail.gradcam.run}
                  </button>
                  {gradcamMsg && (
                    <div
                      style={{
                        padding: "8px 12px",
                        fontFamily: "var(--font-mono)",
                        fontSize: 11,
                        borderRadius: 8,
                        background:
                          gradcamMsg.kind === "error"
                            ? "oklch(0.704 0.191 22.216 / 0.10)"
                            : gradcamMsg.kind === "success"
                              ? "oklch(0.72 0.16 150 / 0.10)"
                              : "rgba(255,255,255,0.04)",
                        color:
                          gradcamMsg.kind === "error"
                            ? "oklch(0.85 0.14 22)"
                            : gradcamMsg.kind === "success"
                              ? "oklch(0.85 0.16 150)"
                              : "var(--vf-text-dim)",
                        wordBreak: "break-all",
                      }}
                    >
                      {gradcamMsg.text}
                    </div>
                  )}
                  {gradcamResult && gradcamResult.items.length > 0 && (
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns:
                          "repeat(auto-fill, minmax(160px, 1fr))",
                        gap: 10,
                      }}
                    >
                      {gradcamResult.items.map((item) => {
                        const url = artifactUrl(item.overlay);
                        return (
                          <button
                            key={item.overlay}
                            type="button"
                            onClick={() =>
                              setLightbox({ src: url, caption: item.source })
                            }
                            style={{
                              background: "rgba(0,0,0,0.3)",
                              // A wrong prediction is the one worth looking at,
                              // so the border says so before the caption does.
                              border:
                                item.correct === false
                                  ? "1px solid oklch(0.74 0.18 22)"
                                  : item.correct === true
                                    ? "1px solid oklch(0.72 0.16 150)"
                                    : "1px solid var(--vf-panel-stroke)",
                              borderRadius: 10,
                              padding: 0,
                              overflow: "hidden",
                              cursor: "zoom-in",
                              display: "flex",
                              flexDirection: "column",
                            }}
                          >
                            <img
                              src={url}
                              alt={item.source}
                              style={{ width: "100%", height: "auto", display: "block" }}
                            />
                            <div
                              style={{
                                padding: "6px 10px 8px",
                                fontFamily: "var(--font-mono)",
                                fontSize: 10,
                                color: "var(--vf-text-dim)",
                                textAlign: "left",
                              }}
                            >
                              <GradCamCaption item={item} />
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              ) : (
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    color: "var(--vf-text-muted)",
                    lineHeight: 1.5,
                  }}
                >
                  {t.runDetail.gradcam.hint}
                </div>
              )}
            </Section>
          )}

          <CrossValidationDetail metrics={detail.metrics} />

          <Section title={t.runDetail.metrics.title}>
            <MetricsGrid metrics={detail.metrics} metricCis={detail.metric_cis} />
          </Section>

          {detail.artifacts.graphics && detail.artifacts.graphics.length > 0 && (
            <Section title={t.runDetail.graphs.title}>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
                  gap: 12,
                }}
              >
                {detail.artifacts.graphics.map((g) => {
                  const filename = g.replace(/\\/g, "/").split("/").pop() ?? g;
                  const label = graphLabels[filename] ?? filename;
                  const url = artifactUrl(g);
                  return (
                    <button
                      key={g}
                      type="button"
                      onClick={() => setLightbox({ src: url, caption: g })}
                      style={{
                        background: "rgba(0,0,0,0.3)",
                        border: "1px solid var(--vf-panel-stroke)",
                        borderRadius: 10,
                        padding: 0,
                        overflow: "hidden",
                        cursor: "zoom-in",
                        display: "flex",
                        flexDirection: "column",
                        gap: 6,
                        textAlign: "left",
                      }}
                    >
                      <img
                        src={url}
                        alt={label}
                        style={{ width: "100%", height: "auto", display: "block" }}
                      />
                      <div
                        style={{
                          padding: "6px 10px 8px",
                          fontFamily: "var(--font-mono)",
                          fontSize: 11,
                          color: "var(--vf-text-dim)",
                        }}
                      >
                        {label}
                        <div
                          style={{
                            fontSize: 9,
                            color: "var(--vf-text-muted)",
                            wordBreak: "break-all",
                            marginTop: 2,
                          }}
                        >
                          {g}
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </Section>
          )}

          <Section
            title={t.runDetail.tests.title}
            action={
              <button
                type="button"
                onClick={() => setShowTestForm((s) => !s)}
                style={{
                  padding: "6px 12px",
                  background: "var(--accent-soft)",
                  border: "1px solid var(--accent-vf)",
                  borderRadius: 8,
                  color: "var(--vf-text)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  cursor: "pointer",
                  letterSpacing: "0.10em",
                  textTransform: "uppercase",
                }}
              >
                {showTestForm ? t.common.cancel : t.runDetail.tests.open}
              </button>
            }
          >
            {showTestForm && (
              <div
                style={{
                  padding: 14,
                  border: "1px dashed var(--vf-panel-stroke)",
                  borderRadius: 10,
                  marginBottom: 12,
                  display: "flex",
                  flexDirection: "column",
                  gap: 10,
                }}
              >
                <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
                  <FormField
                    label={testFolderLabel(t, detail)}
                    value={testForm.data_dir}
                    onChange={(v) => setTestForm((f) => ({ ...f, data_dir: v }))}
                    placeholder={
                      task === "regression"
                        ? t.runDetail.tests.manifestPlaceholder
                        : t.runDetail.tests.folderPlaceholder
                    }
                  />
                  <button
                    type="button"
                    onClick={() => void pickFolder()}
                    style={{
                      padding: "10px 14px",
                      background: "transparent",
                      border: "1px solid var(--vf-panel-stroke)",
                      borderRadius: 10,
                      color: "var(--vf-text-dim)",
                      fontFamily: "var(--font-mono)",
                      fontSize: 11,
                      cursor: "pointer",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {t.runDetail.browse}
                  </button>
                </div>
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 10,
                    lineHeight: 1.7,
                    color: "var(--vf-text-muted)",
                  }}
                >
                  {testFolderHint(t, detail)}
                </div>
                <FormField
                  label={t.runDetail.tests.label}
                  value={testForm.label}
                  onChange={(v) => setTestForm((f) => ({ ...f, label: v }))}
                  placeholder={t.runDetail.tests.labelPlaceholder}
                />
                <button
                  type="button"
                  onClick={() => void runTest()}
                  disabled={testing}
                  style={{
                    padding: "12px 20px",
                    background:
                      "linear-gradient(180deg, var(--accent-soft) 0%, rgba(8,10,14,0.4) 100%)",
                    border: "1px solid var(--accent-vf)",
                    borderRadius: 10,
                    color: "var(--vf-text)",
                    fontFamily: "var(--font-mono)",
                    fontSize: 12,
                    fontWeight: 600,
                    letterSpacing: "0.10em",
                    textTransform: "uppercase",
                    cursor: testing ? "wait" : "pointer",
                    opacity: testing ? 0.6 : 1,
                  }}
                >
                  {testing ? t.runDetail.tests.running : t.runDetail.tests.run}
                </button>
                {testMsg && (
                  <div
                    style={{
                      padding: "8px 12px",
                      fontFamily: "var(--font-mono)",
                      fontSize: 11,
                      borderRadius: 8,
                      background:
                        testMsg.kind === "error"
                          ? "oklch(0.704 0.191 22.216 / 0.10)"
                          : testMsg.kind === "success"
                            ? "oklch(0.72 0.16 150 / 0.10)"
                            : "rgba(255,255,255,0.04)",
                      color:
                        testMsg.kind === "error"
                          ? "oklch(0.85 0.14 22)"
                          : testMsg.kind === "success"
                            ? "oklch(0.85 0.16 150)"
                            : "var(--vf-text-dim)",
                    }}
                  >
                    {testMsg.text}
                  </div>
                )}
              </div>
            )}

            {detail.tests.length === 0 ? (
              <div
                style={{
                  padding: 18,
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  color: "var(--vf-text-muted)",
                  textAlign: "center",
                  border: "1px dashed var(--vf-panel-stroke)",
                  borderRadius: 10,
                }}
              >
                {t.runDetail.tests.empty}
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {detail.tests.map((test) => (
                  <TestRow key={test.test_id} test={test} onOpenImage={(src, caption) => setLightbox({ src, caption })} />
                ))}
              </div>
            )}
          </Section>
        </>
      )}

      {lightbox && (
        <Lightbox
          src={lightbox.src}
          caption={lightbox.caption}
          onClose={() => setLightbox(null)}
        />
      )}
    </div>
  );
}

interface CVFold {
  fold: number;
  train_size: number;
  val_size: number;
  status: string;
  error?: string;
  best_val_loss: number | null;
  accuracy: number | null;
  f1: number | null;
}

interface CVAggregate {
  n_folds: number;
  n_folds_ok: number;
  n_folds_failed: number;
  /** Folds the server cut when a stop arrived (ADR-111); absent on older runs. */
  n_folds_stopped?: number;
  mean_accuracy: number | null;
  std_accuracy: number | null;
  mean_f1: number | null;
  std_f1: number | null;
  /** The divisor of the std: 1 is the sample std (n−1). Absent on runs from before
   *  ADR-111, which divided by n. */
  std_ddof?: number;
}

/** Per-fold detail section, only rendered when the run.json carries
 *  ``fold_results`` (i.e. it was produced by CrossValidationBlock). */
function CrossValidationDetail({ metrics }: { metrics: Record<string, unknown> }) {
  const t = useT();
  const folds = metrics["fold_results"];
  const agg = metrics["cv_aggregate"];
  if (!Array.isArray(folds) || folds.length === 0) return null;

  const typed = folds as CVFold[];
  const a = (agg ?? {}) as CVAggregate;

  return (
    <Section
      title={t.runDetail.cv.title(
        a.n_folds_ok ?? typed.length,
        a.n_folds ?? typed.length,
        a.n_folds_failed ?? 0,
        a.n_folds_stopped ?? 0,
      )}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: 10,
          marginBottom: 12,
        }}
      >
        {a.mean_accuracy !== null && a.mean_accuracy !== undefined && (
          <CVAggregateCard
            label={t.runDetail.cv.meanAccuracy}
            mean={a.mean_accuracy}
            std={a.std_accuracy ?? null}
            n={a.n_folds_ok}
            highlight
          />
        )}
        {a.mean_f1 !== null && a.mean_f1 !== undefined && (
          <CVAggregateCard
            label={t.runDetail.cv.meanF1}
            mean={a.mean_f1}
            std={a.std_f1 ?? null}
            n={a.n_folds_ok}
          />
        )}
      </div>
      {/* A std is only comparable between runs that divide alike. */}
      {a.mean_accuracy !== null && a.mean_accuracy !== undefined && (
        <div
          style={{
            marginBottom: 12,
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            color: "var(--vf-text-muted)",
          }}
        >
          {t.runDetail.cv.stdNote(stdDdof(a))}
        </div>
      )}

      <div
        style={{
          padding: 10,
          background: "rgba(0,0,0,0.30)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 10,
          overflowX: "auto",
        }}
      >
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
          }}
        >
          <thead>
            <tr>
              <th style={cvThStyle}>{t.runDetail.cv.fold}</th>
              <th style={cvThStyle}>{t.runDetail.cv.train}</th>
              <th style={cvThStyle}>{t.runDetail.cv.val}</th>
              <th style={cvThStyle}>{t.runDetail.cv.valLoss}</th>
              <th style={cvThStyle}>{t.runDetail.cv.accuracy}</th>
              <th style={cvThStyle}>F1</th>
              <th style={cvThStyle}>{t.runDetail.cv.status}</th>
            </tr>
          </thead>
          <tbody>
            {typed.map((f) => {
              const state = unitState(f.status);
              return (
                <tr key={f.fold}>
                  <td style={cvTdLabelStyle}>#{f.fold + 1}</td>
                  <td style={cvTdStyle}>{f.train_size}</td>
                  <td style={cvTdStyle}>{f.val_size}</td>
                  <td style={cvTdStyle}>{fmtMetric(f.best_val_loss)}</td>
                  <td style={cvTdStyle}>{fmtMetric(f.accuracy)}</td>
                  <td style={cvTdStyle}>{fmtMetric(f.f1)}</td>
                  <td
                    style={{
                      ...cvTdStyle,
                      color:
                        state === "ok"
                          ? "oklch(0.85 0.16 150)"
                          : state === "stopped"
                            ? STOPPED_COLOR
                            : "oklch(0.85 0.14 22)",
                    }}
                  >
                    {state === "ok"
                      ? "✓"
                      : state === "stopped"
                        ? `■ ${t.resultsView.outcome.stopped}`
                        : `× ${f.error || "?"}`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

function CVAggregateCard({
  label,
  mean,
  std,
  n,
  highlight,
}: {
  label: string;
  mean: number;
  /** Null below two finished folds: one fold has no spread, not a spread of zero. */
  std: number | null;
  n?: number;
  highlight?: boolean;
}) {
  const t = useT();
  const accent = "var(--accent-vf)";
  return (
    <div
      style={{
        padding: 14,
        borderRadius: 10,
        border: "1px solid var(--vf-panel-stroke)",
        background: highlight
          ? "linear-gradient(180deg, var(--accent-soft) 0%, rgba(12,14,18,0.5) 100%)"
          : "rgba(12,14,18,0.55)",
      }}
    >
      <div
        style={{
          fontSize: 9,
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
          fontFamily: "var(--font-mono)",
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 20,
          marginTop: 6,
          fontFamily: "var(--font-mono)",
          fontWeight: 600,
          color: highlight ? accent : "var(--vf-text)",
        }}
      >
        {mean.toFixed(4)}
        <span
          style={{
            fontSize: 12,
            color: "var(--vf-text-muted)",
            marginLeft: 6,
            fontWeight: 400,
          }}
        >
          ± {fmtMetric(std)}
          {typeof n === "number" && ` · ${t.resultsView.taskCv.sample(n)}`}
        </span>
      </div>
    </div>
  );
}

const cvThStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "6px 8px",
  borderBottom: "1px solid var(--vf-panel-stroke)",
  fontSize: 9,
  letterSpacing: "0.14em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
  fontWeight: 500,
};

const cvTdStyle: React.CSSProperties = {
  padding: "6px 8px",
  borderBottom: "1px solid rgba(255,255,255,0.04)",
  color: "var(--vf-text)",
};

const cvTdLabelStyle: React.CSSProperties = {
  ...cvTdStyle,
  color: "var(--vf-text-muted)",
  fontSize: 10,
};


interface SectionProps {
  title: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}

function Section({ title, children, action }: SectionProps) {
  return (
    <div
      style={{
        padding: "14px 16px",
        background: "rgba(255,255,255,0.02)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 12,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 10,
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
        {action}
      </div>
      {children}
    </div>
  );
}

const PATH_ROW_BUTTON_STYLE: CSSProperties = {
  padding: "4px 8px",
  background: "rgba(255,255,255,0.04)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 6,
  color: "var(--vf-text-dim)",
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  cursor: "pointer",
};

function PathRow({
  label,
  value,
  action,
}: {
  label: string;
  value: string;
  /** Extra button beside "copy", e.g. open the folder. */
  action?: ReactNode;
}) {
  const t = useT();
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "6px 0",
        borderTop: "1px solid var(--vf-panel-stroke)",
        marginTop: 6,
      }}
    >
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--vf-text-muted)",
          letterSpacing: "0.10em",
          textTransform: "uppercase",
          minWidth: 110,
        }}
      >
        {label}
      </span>
      <code
        style={{
          flex: 1,
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          color: "var(--vf-text)",
          wordBreak: "break-all",
        }}
      >
        {value}
      </code>
      <button
        type="button"
        onClick={() => void navigator.clipboard.writeText(value)}
        title={t.runDetail.copyPath}
        style={PATH_ROW_BUTTON_STYLE}
      >
        {t.runDetail.copy}
      </button>
      {action}
    </div>
  );
}

function KeyRow({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        display: "flex",
        gap: 12,
        padding: "6px 0",
        borderTop: "1px solid var(--vf-panel-stroke)",
        marginTop: 6,
      }}
    >
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--vf-text-muted)",
          letterSpacing: "0.10em",
          textTransform: "uppercase",
          minWidth: 110,
        }}
      >
        {label}
      </span>
      <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--vf-text)" }}>
        {value}
      </span>
    </div>
  );
}

/** The task a run belongs to, from its run.json (mirrors the backend's rule). */
function runTask(detail: RunDetail | null): string {
  const config = (detail?.config ?? {}) as { task?: string };
  const task = config.task;
  if (task && ["regression", "segmentation", "anomaly", "detection"].includes(task)) {
    return task;
  }
  return "classification";
}

/** What the single test input is called for this task (ADR-080). */
function testFolderLabel(t: Dict, detail: RunDetail | null): string {
  return runTask(detail) === "regression"
    ? t.runDetail.tests.manifest
    : t.runDetail.tests.folder;
}

/** What that folder has to contain — the label shape the run was trained with.
 *
 * Spelled out per task because "one folder" is only unambiguous once you know
 * where the labels live, and getting it wrong is a failed evaluation rather
 * than a validation error.
 */
function testFolderHint(t: Dict, detail: RunDetail | null): string {
  const hints = t.runDetail.tests.folderHint;
  switch (runTask(detail)) {
    case "detection":
      return hints.detection;
    case "segmentation":
      return hints.segmentation;
    case "anomaly":
      return hints.anomaly;
    case "regression":
      return hints.regression;
    default:
      return hints.classification;
  }
}

/** Caption under a Grad-CAM overlay: what the model said, and what it should have.
 *
 * The ground truth only exists when the images came from class-named folders, so
 * the "real" line is omitted rather than filled with a guess (ADR-077).
 */
function GradCamCaption({ item }: { item: GradCamItem }) {
  const t = useT();
  const predicted =
    item.predicted_label ??
    (item.predicted_class !== null ? t.runDetail.gradcam.classNumber(item.predicted_class) : null);

  if (item.true_class === null || item.true_class === undefined) {
    return <>{item.prediction ?? (predicted ? t.runDetail.gradcam.predictedIs(predicted) : "—")}</>;
  }
  return (
    <>
      <div>
        <span style={{ color: "var(--vf-text-muted)" }}>{t.runDetail.gradcam.actual}</span> {item.true_class}
      </div>
      <div style={{ color: item.correct ? "inherit" : "oklch(0.8 0.16 22)" }}>
        <span style={{ color: "var(--vf-text-muted)" }}>{t.runDetail.gradcam.predicted}</span> {predicted}
        {item.correct === false ? " ✗" : " ✓"}
      </div>
    </>
  );
}

function MetricsGrid({
  metrics,
  metricCis,
}: {
  metrics: Record<string, unknown>;
  metricCis?: Record<string, MetricCI>;
}) {
  const t = useT();
  const entries = Object.entries(metrics);
  if (entries.length === 0) {
    return <div style={{ color: "var(--vf-text-muted)", fontSize: 12 }}>{t.runDetail.metrics.none}</div>;
  }
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
        gap: 8,
      }}
    >
      {entries.map(([k, v]) => {
        const ci = metricCi(metricCis, k);
        return (
          <div
            key={k}
            style={{
              padding: "8px 10px",
              background: "rgba(0,0,0,0.3)",
              border: "1px solid var(--vf-panel-stroke)",
              borderRadius: 8,
            }}
          >
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 9,
                letterSpacing: "0.14em",
                textTransform: "uppercase",
                color: "var(--vf-text-muted)",
              }}
            >
              {metricLabel(t, k)}
            </div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 15,
                fontWeight: 600,
                color: "var(--vf-text)",
                marginTop: 2,
              }}
            >
              {fmtMetric(v)}
            </div>
            {ci && (
              <div
                title={t.metrics.ciTooltip(
                  Math.round(ci.confidence * 100),
                  ci.n_resamples,
                  ci.n_samples,
                )}
                style={{
                  marginTop: 3,
                  fontFamily: "var(--font-mono)",
                  fontSize: 9,
                  color: "var(--vf-text-muted)",
                  whiteSpace: "nowrap",
                  cursor: "help",
                }}
              >
                {ci.ci_low.toFixed(4)} – {ci.ci_high.toFixed(4)}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

interface TestRowProps {
  test: TestRecord;
  onOpenImage: (src: string, caption: string) => void;
}

function TestRow({ test, onOpenImage }: TestRowProps) {
  const { locale } = useI18n();
  return (
    <div
      style={{
        padding: "10px 14px",
        background: "rgba(255,255,255,0.025)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 10,
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontWeight: 600, fontSize: 13 }}>{test.label}</span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--vf-text-muted)" }}>
          {new Date(test.timestamp).toLocaleString(locale)}
        </span>
      </div>
      <code
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--vf-text-dim)",
          wordBreak: "break-all",
        }}
      >
        {test.base_dir}
      </code>
      <MetricsGrid metrics={test.metrics} />
      {Object.entries(test.artifacts).length > 0 && (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {Object.entries(test.artifacts).map(([k, p]) => (
            <button
              key={k}
              type="button"
              onClick={() => onOpenImage(artifactUrl(p), p)}
              style={{
                padding: "4px 10px",
                background: "rgba(255,255,255,0.04)",
                border: "1px solid var(--vf-panel-stroke)",
                borderRadius: 6,
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                color: "var(--vf-text-dim)",
                cursor: "pointer",
              }}
            >
              📊 {k}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

interface FormFieldProps {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}

/** The hyperparameters this run was trained with — reproducibility at a glance.
 *
 * Reads the task-agnostic ``training`` block (every task config carries one) and
 * the optional ``transfer_learning`` field (regression/segmentation, ADR-046/047).
 * Only scalar knobs are shown; nested config (scheduler) is skipped.
 */
function TrainingSection({ config }: { config: Record<string, unknown> }) {
  const t = useT();
  const training = getConfigRecord(config, "training");
  if (!training) return null;

  const KNOBS = [
    "learning_rate",
    "epochs",
    "batch_size",
    "optimizer",
    "loss",
    "weight_decay",
    "early_stopping_patience",
    "seed",
    // A seed alone does not make a run reproducible on GPU; whether cuDNN was
    // pinned is part of the same claim, so it belongs next to it.
    "deterministic",
  ];
  const rows = KNOBS.filter(
    (k) => training[k] !== undefined && training[k] !== null,
  ).map((k) => [k, training[k]] as const);

  const tl = getConfigRecord(config, "transfer_learning");
  if (rows.length === 0 && !tl) return null;

  return (
    <Section title={t.runDetail.training.title}>
      {rows.map(([k, v]) => (
        <KeyRow key={k} label={k.replace(/_/g, " ")} value={String(v)} />
      ))}
      {tl && (
        <KeyRow
          label={t.runDetail.training.transferLearning}
          value={
            tl["mode"] === "fine_tuning"
              ? `fine-tuning · backbone lr × ${String(tl["backbone_lr_multiplier"])}`
              : String(tl["mode"])
          }
        />
      )}
    </Section>
  );
}

/** Pre-training pipeline that produced this run — preprocessing + augmentation.
 *
 * Reproducibility is the whole point of saving run.json. The training-time
 * filter pipeline and augmentation flags are the bits most likely to be
 * forgotten by the researcher months later, so we surface them up front.
 */
function PipelineSection({ config }: { config: Record<string, unknown> }) {
  const t = useT();
  const data = getDataSection(config);
  if (!data) return null;

  const ppSteps = Array.isArray(data.preprocessing?.steps)
    ? data.preprocessing!.steps!
    : [];
  const transforms = data.transforms ?? {};
  const transformEntries = Object.entries(transforms).filter(
    ([, v]) => v !== null && v !== undefined && v !== "",
  );

  if (ppSteps.length === 0 && transformEntries.length === 0) return null;

  return (
    <Section title={t.runDetail.pipeline.title}>
      {ppSteps.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 12 }}>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 9,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              color: "var(--vf-text-muted)",
            }}
          >
            {t.runDetail.pipeline.preprocessing}
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {ppSteps.map((step, i) => {
              const kind = String(step["kind"] ?? "");
              const params = Object.entries(step).filter(([k]) => k !== "kind");
              const label = PREPROCESS_KIND_LABELS[kind] ?? kind;
              const paramStr = params
                .map(([k, v]) => `${k}=${typeof v === "number" ? v : JSON.stringify(v)}`)
                .join(", ");
              return (
                <span
                  key={`${kind}-${i}`}
                  style={{
                    padding: "5px 10px",
                    background: "rgba(0,0,0,0.30)",
                    border: "1px solid var(--vf-panel-stroke)",
                    borderRadius: 8,
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    color: "var(--vf-text-dim)",
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 8,
                  }}
                >
                  <span style={{ color: "var(--accent-vf)" }}>{i + 1}.</span>
                  <span style={{ color: "var(--vf-text)" }}>{label}</span>
                  {paramStr && (
                    <span style={{ color: "var(--vf-text-muted)", fontSize: 10 }}>
                      ({paramStr})
                    </span>
                  )}
                </span>
              );
            })}
          </div>
        </div>
      )}

      {transformEntries.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 9,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              color: "var(--vf-text-muted)",
            }}
          >
            {t.runDetail.pipeline.augmentation}
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
              gap: 6,
            }}
          >
            {transformEntries.map(([k, v]) => (
              <div
                key={k}
                style={{
                  padding: "6px 10px",
                  background: "rgba(0,0,0,0.25)",
                  border: "1px solid var(--vf-panel-stroke)",
                  borderRadius: 6,
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  gap: 8,
                }}
              >
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 10,
                    color: "var(--vf-text-muted)",
                  }}
                >
                  {k}
                </span>
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    color: "var(--vf-text)",
                    wordBreak: "break-all",
                    textAlign: "right",
                  }}
                >
                  {formatTransformValue(v)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </Section>
  );
}

function formatTransformValue(v: unknown): string {
  if (v === true) return "✓";
  if (v === false) return "—";
  if (Array.isArray(v)) return `[${v.map(String).join(", ")}]`;
  return String(v);
}

const exportLabelStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 4,
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  color: "var(--vf-text-muted)",
  letterSpacing: "0.10em",
  textTransform: "uppercase",
};

const exportInputStyle: React.CSSProperties = {
  padding: "8px 10px",
  background: "rgba(0,0,0,0.35)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 8,
  color: "var(--vf-text)",
  fontFamily: "var(--font-mono)",
  fontSize: 12,
};

function ExportToggle({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <label
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        fontFamily: "var(--font-mono)",
        fontSize: 11,
        color: value ? "var(--vf-text)" : "var(--vf-text-dim)",
        cursor: "pointer",
      }}
    >
      <input
        type="checkbox"
        checked={value}
        onChange={(e) => onChange(e.target.checked)}
        style={{ accentColor: "var(--accent-vf)" }}
      />
      {label}
    </label>
  );
}

function ExportResultPanel({ result }: { result: ExportOnnxResponse }) {
  const t = useT();
  const stats = t.runDetail.onnx.stats;
  const sizeMb = (result.file_size_bytes / (1024 * 1024)).toFixed(2);
  const val = result.validation;
  const bench = result.benchmark;
  return (
    <div
      style={{
        marginTop: 12,
        padding: 12,
        background: "rgba(0,0,0,0.30)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 10,
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
        gap: 10,
        fontFamily: "var(--font-mono)",
      }}
    >
      <ExportStat label={stats.fileSize} value={`${sizeMb} MB`} />
      {val && (
        <ExportStat
          label={stats.maxDiff}
          value={
            typeof val.max_diff === "number" ? val.max_diff.toExponential(3) : "—"
          }
          accent={val.within_tolerance ? "oklch(0.85 0.16 150)" : "oklch(0.85 0.14 22)"}
        />
      )}
      {bench && (
        <>
          <ExportStat
            label={stats.onnxLatency}
            value={
              typeof bench.mean_ms === "number"
                ? `${bench.mean_ms.toFixed(2)} ms`
                : "—"
            }
          />
          <ExportStat
            label={stats.onnxP95}
            value={
              typeof bench.p95_ms === "number" ? `${bench.p95_ms.toFixed(2)} ms` : "—"
            }
          />
          <ExportStat
            label={stats.torchLatency}
            value={
              typeof bench.torch_mean_ms === "number"
                ? `${bench.torch_mean_ms.toFixed(2)} ms`
                : "—"
            }
          />
          <ExportStat
            label={stats.speedup}
            value={
              typeof bench.speedup === "number" ? `${bench.speedup.toFixed(2)}×` : "—"
            }
            accent={
              typeof bench.speedup === "number" && bench.speedup >= 1
                ? "oklch(0.85 0.16 150)"
                : undefined
            }
          />
          <ExportStat label={stats.runs} value={String(bench.runs ?? "—")} />
        </>
      )}
    </div>
  );
}

function BatchResultPanel({ result }: { result: BatchPredictResponse }) {
  const t = useT();
  const failedHead = result.failed_files.slice(0, 5);
  const more = result.failed_files.length - failedHead.length;
  return (
    <div
      style={{
        marginTop: 12,
        padding: 12,
        background: "rgba(0,0,0,0.30)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 10,
        display: "flex",
        flexDirection: "column",
        gap: 10,
        fontFamily: "var(--font-mono)",
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
          gap: 10,
        }}
      >
        <ExportStat
          label={t.runDetail.batch.processed}
          value={String(result.total_processed)}
          accent="oklch(0.85 0.16 150)"
        />
        <ExportStat
          label={t.runDetail.batch.failedCount}
          value={String(result.failed_count)}
          accent={
            result.failed_count === 0
              ? "var(--vf-text)"
              : "oklch(0.85 0.14 22)"
          }
        />
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <span
          style={{
            fontSize: 10,
            color: "var(--vf-text-muted)",
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            minWidth: 70,
          }}
        >
          {t.runDetail.batch.csv}
        </span>
        <code
          style={{
            flex: 1,
            fontSize: 11,
            color: "var(--vf-text)",
            wordBreak: "break-all",
          }}
        >
          {result.output_csv}
        </code>
        <button
          type="button"
          onClick={() => void navigator.clipboard.writeText(result.output_csv)}
          title={t.runDetail.copyPath}
          style={{
            padding: "4px 8px",
            background: "rgba(255,255,255,0.04)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 6,
            color: "var(--vf-text-dim)",
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            cursor: "pointer",
          }}
        >
          {t.runDetail.copy}
        </button>
      </div>
      {failedHead.length > 0 && (
        <details style={{ fontSize: 11, color: "var(--vf-text-muted)" }}>
          <summary style={{ cursor: "pointer", color: "oklch(0.85 0.14 22)" }}>
            {t.runDetail.batch.failedFiles(result.failed_count)}
          </summary>
          <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
            {failedHead.map((p) => (
              <li key={p} style={{ wordBreak: "break-all" }}>
                {p}
              </li>
            ))}
          </ul>
          {more > 0 && (
            <div style={{ marginTop: 4, fontSize: 10 }}>
              {t.runDetail.batch.more(more)}
            </div>
          )}
        </details>
      )}
    </div>
  );
}

function ExportStat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: string;
}) {
  return (
    <div
      style={{
        padding: "6px 10px",
        background: "rgba(255,255,255,0.025)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 6,
      }}
    >
      <div
        style={{
          fontSize: 9,
          color: "var(--vf-text-muted)",
          letterSpacing: "0.14em",
          textTransform: "uppercase",
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: 13,
          fontWeight: 600,
          color: accent ?? "var(--vf-text)",
          marginTop: 2,
          wordBreak: "break-all",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function FormField({ label, value, onChange, placeholder }: FormFieldProps) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1 }}>
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 9,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
        }}
      >
        {label}
      </span>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        style={{
          padding: "8px 10px",
          background: "rgba(0,0,0,0.35)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 8,
          color: "var(--vf-text)",
          fontFamily: "var(--font-mono)",
          fontSize: 12,
        }}
      />
    </label>
  );
}
