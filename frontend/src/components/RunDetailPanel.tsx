import { useEffect, useState } from "react";
import {
  ApiError,
  artifactUrl,
  batchPredictRun,
  downloadRunMarkdown,
  exportRunToOnnx,
  fetchRunDetail,
  gradcamRun,
  pickDatasetFolder,
  testRunOnDataset,
  type BatchPredictResponse,
  type ExportOnnxResponse,
  type GradCamResponse,
  type RunDetail,
  type TestRecord,
} from "../api/client";
import { Lightbox } from "./Lightbox";

interface RunDetailPanelProps {
  runId: string;
  onBack: () => void;
}

/** Plot file naming convention from the backend; used to humanize labels. */
const GRAPH_LABELS: Record<string, string> = {
  "loss.png": "Loss (train + val)",
  "accuracy.png": "Accuracy (train + val)",
  "confusion_matrix.png": "Matriz de confusão",
  "confusion_matrix_normalized.png": "Matriz de confusão (normalizada)",
  "roc_curve.png": "Curva ROC",
  "precision_recall_curve.png": "Curva Precision-Recall",
  // Detection (Ultralytics / torchvision) plot names.
  "results.png": "Resultados (loss + mAP)",
  "BoxPR_curve.png": "Curva Precision-Recall (box)",
  "BoxF1_curve.png": "Curva F1 (box)",
};

function metricLabel(key: string): string {
  const labels: Record<string, string> = {
    accuracy: "Acurácia",
    f1: "F1",
    precision: "Precisão",
    recall: "Recall",
    auc_roc: "AUC-ROC",
    test_accuracy: "Acurácia (teste)",
    test_f1: "F1 (teste)",
    test_precision: "Precisão (teste)",
    test_recall: "Recall (teste)",
    test_auc_roc: "AUC-ROC (teste)",
    best_val_loss: "Melhor val loss",
    best_epoch: "Melhor epoch",
    total_epochs: "Epochs treinados",
    // Detection metrics (mAP @ IoU thresholds; box validation loss).
    map50: "mAP@50",
    map50_95: "mAP@50-95",
    box_loss: "Box loss (val)",
  };
  return labels[key] ?? key;
}

function fmtMetric(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return v % 1 === 0 ? String(v) : v.toFixed(4);
  return String(v);
}

/** Human-friendly name for a preprocessing filter kind. */
const PREPROCESS_KIND_LABELS: Record<string, string> = {
  gaussian_blur: "Gaussian blur",
  median_blur: "Median blur",
  unsharp: "Unsharp mask",
  edges: "Edges (Sobel)",
  emboss: "Emboss",
  grayscale: "Grayscale",
  equalize: "Equalize (CLAHE)",
  autocontrast: "Autocontrast",
  wavelet: "Wavelet (Haar)",
};

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
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lightbox, setLightbox] = useState<{ src: string; caption: string } | null>(null);
  const [testForm, setTestForm] = useState({
    base_dir: "",
    train_dir: "train",
    val_dir: "val",
    test_dir: "test",
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

  // Grad-CAM state — independent single-shot explainability action.
  const [showGradcamForm, setShowGradcamForm] = useState(false);
  const [gradcamForm, setGradcamForm] = useState({ input_dir: "", num_samples: 8 });
  const [gradcamRunning, setGradcamRunning] = useState(false);
  const [gradcamResult, setGradcamResult] = useState<GradCamResponse | null>(null);
  const [gradcamMsg, setGradcamMsg] = useState<{ kind: "info" | "error" | "success"; text: string } | null>(null);

  // A detection run.json carries task="detection". Some post-training actions
  // (batch CSV inference, per-model evaluate) are classification-only and hidden
  // for detection runs (see PHASE7_DETECTION_PLAN brick D/F).
  const runTask = detail?.config?.["task"] as string | undefined;
  const isDetection = runTask === "detection";
  const detectionBackend = (
    detail?.config?.["model"] as Record<string, unknown> | undefined
  )?.["backend"] as string | undefined;
  // ONNX export: classification + regression + segmentation (shared core helper)
  // and Ultralytics detection (torchvision detection export is not implemented).
  // Anomaly is excluded — PatchCore's memory-bank scoring has no forward graph.
  const canExportOnnx = isDetection
    ? detectionBackend === "ultralytics"
    : runTask !== "anomaly";
  // Batch CSV inference: classification + regression (continuous targets) +
  // anomaly (score + threshold decision). Detection/segmentation produce
  // per-box/per-pixel outputs that don't map to a flat row yet (ADR-041 slice 3).
  const canBatchPredict = !isDetection && runTask !== "segmentation";
  // Grad-CAM: classification + regression + segmentation (conv-based CAM, ADR-053).
  // Detection (Ultralytics) and anomaly (no class/conv target) are excluded.
  const canGradcam = !isDetection && runTask !== "anomaly";

  useEffect(() => {
    let alive = true;
    setLoading(true);
    fetchRunDetail(runId)
      .then((d) => {
        if (alive) setDetail(d);
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setError(e instanceof Error ? e.message : "Falha ao carregar detalhes.");
      })
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [runId]);

  const reload = async () => {
    try {
      const d = await fetchRunDetail(runId);
      setDetail(d);
    } catch {
      /* ignore — keep existing detail */
    }
  };

  const pickFolder = async () => {
    setTestMsg({ kind: "info", text: "Abrindo seletor…" });
    try {
      const res = await pickDatasetFolder();
      if (res.cancelled) {
        setTestMsg({ kind: "info", text: res.message ?? "Cancelado." });
        return;
      }
      setTestForm((f) => ({ ...f, base_dir: res.path }));
      setTestMsg({ kind: "success", text: `Pasta: ${res.path}` });
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Falha ao escolher pasta.";
      setTestMsg({ kind: "error", text: msg });
    }
  };

  const pickBatchFolder = async () => {
    setBatchMsg({ kind: "info", text: "Abrindo seletor…" });
    try {
      const res = await pickDatasetFolder();
      if (res.cancelled) {
        setBatchMsg({ kind: "info", text: res.message ?? "Cancelado." });
        return;
      }
      setBatchForm((f) => ({ ...f, input_dir: res.path }));
      setBatchMsg({ kind: "success", text: `Pasta: ${res.path}` });
    } catch (e) {
      setBatchMsg({
        kind: "error",
        text: e instanceof Error ? e.message : "Falha ao escolher pasta.",
      });
    }
  };

  const runBatch = async () => {
    if (!batchForm.input_dir.trim()) {
      setBatchMsg({
        kind: "error",
        text: "Informe a pasta de imagens para inferência.",
      });
      return;
    }
    setBatchRunning(true);
    setBatchResult(null);
    setBatchMsg({ kind: "info", text: "Rodando inferência em lote…" });
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
            ? `${okCount} imagens processadas · CSV em ${result.output_csv}`
            : `${okCount} ok · ${failed} falharam · CSV em ${result.output_csv}`,
      });
    } catch (e) {
      const msg =
        e instanceof ApiError
          ? e.message
          : e instanceof Error
            ? e.message
            : "Falha na inferência em lote.";
      setBatchMsg({ kind: "error", text: msg });
    } finally {
      setBatchRunning(false);
    }
  };

  const pickGradcamFolder = async () => {
    setGradcamMsg({ kind: "info", text: "Abrindo seletor…" });
    try {
      const res = await pickDatasetFolder();
      if (res.cancelled) {
        setGradcamMsg({ kind: "info", text: res.message ?? "Cancelado." });
        return;
      }
      setGradcamForm((f) => ({ ...f, input_dir: res.path }));
      setGradcamMsg({ kind: "success", text: `Pasta: ${res.path}` });
    } catch (e) {
      setGradcamMsg({
        kind: "error",
        text: e instanceof Error ? e.message : "Falha ao escolher pasta.",
      });
    }
  };

  const runGradcam = async () => {
    if (!gradcamForm.input_dir.trim()) {
      setGradcamMsg({ kind: "error", text: "Informe a pasta de imagens." });
      return;
    }
    setGradcamRunning(true);
    setGradcamResult(null);
    setGradcamMsg({ kind: "info", text: "Gerando mapas Grad-CAM…" });
    try {
      const result = await gradcamRun(runId, {
        input_dir: gradcamForm.input_dir,
        num_samples: gradcamForm.num_samples,
      });
      setGradcamResult(result);
      setGradcamMsg({
        kind: "success",
        text: `${result.count} mapa(s) gerado(s) · camada ${result.target_layer}`,
      });
    } catch (e) {
      const msg =
        e instanceof ApiError
          ? e.message
          : e instanceof Error
            ? e.message
            : "Falha ao gerar Grad-CAM.";
      setGradcamMsg({ kind: "error", text: msg });
    } finally {
      setGradcamRunning(false);
    }
  };

  const runExport = async () => {
    setExporting(true);
    setExportMsg({ kind: "info", text: "Exportando para ONNX…" });
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
        text: `ONNX salvo em ${result.output_onnx}`,
      });
    } catch (e) {
      const msg =
        e instanceof ApiError
          ? e.message
          : e instanceof Error
            ? e.message
            : "Falha ao exportar ONNX.";
      setExportMsg({ kind: "error", text: msg });
    } finally {
      setExporting(false);
    }
  };

  const runTest = async () => {
    if (!testForm.base_dir.trim()) {
      setTestMsg({ kind: "error", text: "Informe o diretório base do dataset de teste." });
      return;
    }
    setTesting(true);
    setTestMsg({ kind: "info", text: "Avaliando modelo no novo dataset…" });
    try {
      const record = await testRunOnDataset(runId, {
        base_dir: testForm.base_dir,
        train_dir: testForm.train_dir || "train",
        val_dir: testForm.val_dir || "val",
        test_dir: testForm.test_dir || "test",
        label: testForm.label || undefined,
      });
      setTestMsg({
        kind: "success",
        text: `Teste registrado: ${record.test_id}`,
      });
      setShowTestForm(false);
      await reload();
    } catch (e) {
      const msg =
        e instanceof ApiError ? e.message : e instanceof Error ? e.message : "Falha no teste.";
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
          ← histórico
        </button>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: 14, color: "var(--vf-text)" }}>
          {runId}
        </div>
        <button
          type="button"
          onClick={() => void downloadRunMarkdown(runId)}
          title="Baixar model card (markdown) deste run"
          style={{
            marginLeft: "auto",
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
          ↓ markdown
        </button>
      </div>

      {loading && (
        <div style={{ padding: 32, textAlign: "center", color: "var(--vf-text-muted)" }}>
          carregando…
        </div>
      )}

      {error && (
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
          {error}
        </div>
      )}

      {detail && (
        <>
          <Section title="Localização no disco">
            <PathRow label="Pasta do run" value={detail.run_dir} />
            {detail.artifacts.model && (
              <PathRow label="Checkpoint" value={detail.artifacts.model} />
            )}
            {detail.device_used && (
              <KeyRow label="Dispositivo usado" value={detail.device_used} />
            )}
            {detail.environment &&
              Object.entries(detail.environment).map(([k, v]) => (
                <KeyRow key={k} label={`env · ${k}`} value={v} />
              ))}
          </Section>

          <TrainingSection config={detail.config} />

          <PipelineSection config={detail.config} />

          {detail.artifacts.model && canExportOnnx && (
            <Section
              title="Exportar para ONNX"
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
                  {showExportForm ? "cancelar" : "↗ exportar onnx"}
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
                      <span>opset_version</span>
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
                      <span>benchmark_runs</span>
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
                      label="dynamic_axes"
                      value={exportForm.dynamic_axes}
                      onChange={(v) =>
                        setExportForm((f) => ({ ...f, dynamic_axes: v }))
                      }
                    />
                    <ExportToggle
                      label="validate"
                      value={exportForm.validate}
                      onChange={(v) =>
                        setExportForm((f) => ({ ...f, validate: v }))
                      }
                    />
                    <ExportToggle
                      label="benchmark"
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
                    {exporting ? "Exportando…" : "▶ Rodar export"}
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
                  Converte o checkpoint para ONNX, valida diff numérico contra
                  PyTorch e mede latência de inferência. O arquivo é salvo ao
                  lado do checkpoint como <code>best_model.onnx</code>.
                </div>
              )}
              {exportResult && (
                <ExportResultPanel result={exportResult} />
              )}
            </Section>
          )}

          {detail.artifacts.model && canBatchPredict && (
            <Section
              title="Inferência em lote (CSV)"
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
                  {showBatchForm ? "cancelar" : "+ inferência em lote"}
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
                      label="Pasta de imagens"
                      value={batchForm.input_dir}
                      onChange={(v) =>
                        setBatchForm((f) => ({ ...f, input_dir: v }))
                      }
                      placeholder="ex: C:/datasets/inbox"
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
                      📁 Escolher
                    </button>
                  </div>
                  <ExportToggle
                    label="recursive (subpastas)"
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
                    {batchRunning ? "Processando…" : "▶ Rodar inferência"}
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
                  Roda o checkpoint sobre uma pasta de imagens e escreve um CSV
                  com uma linha por imagem (probabilidades + classe predita).
                  Útil para classificar batches de dados novos sem retreinar.
                </div>
              )}
              {batchResult && <BatchResultPanel result={batchResult} />}
            </Section>
          )}

          {detail.artifacts.model && canGradcam && (
            <Section
              title="Grad-CAM (explicabilidade)"
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
                  {showGradcamForm ? "cancelar" : "🔥 Grad-CAM"}
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
                      label="Pasta de imagens"
                      value={gradcamForm.input_dir}
                      onChange={(v) =>
                        setGradcamForm((f) => ({ ...f, input_dir: v }))
                      }
                      placeholder="ex: C:/datasets/amostras"
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
                      📁 Escolher
                    </button>
                  </div>
                  <FormField
                    label="Nº de amostras (1–64)"
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
                    {gradcamRunning ? "Gerando…" : "▶ Gerar Grad-CAM"}
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
                              border: "1px solid var(--vf-panel-stroke)",
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
                              }}
                            >
                              {item.prediction ??
                                `classe predita: ${item.predicted_class}`}
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
                  Gera mapas de calor Grad-CAM sobre imagens de exemplo,
                  destacando as regiões que mais influenciaram a classe predita
                  pelo checkpoint. Útil para interpretar o que o modelo aprendeu.
                </div>
              )}
            </Section>
          )}

          <CrossValidationDetail metrics={detail.metrics} />

          <Section title="Métricas">
            <MetricsGrid metrics={detail.metrics} />
          </Section>

          {detail.artifacts.graphics && detail.artifacts.graphics.length > 0 && (
            <Section title="Gráficos (clique para expandir)">
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
                  gap: 12,
                }}
              >
                {detail.artifacts.graphics.map((g) => {
                  const filename = g.replace(/\\/g, "/").split("/").pop() ?? g;
                  const label = GRAPH_LABELS[filename] ?? filename;
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
            title="Testes neste modelo"
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
                {showTestForm ? "cancelar" : "+ testar"}
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
                    label="Diretório base"
                    value={testForm.base_dir}
                    onChange={(v) => setTestForm((f) => ({ ...f, base_dir: v }))}
                    placeholder="ex: C:/datasets/coffee_v2"
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
                    📁 Escolher
                  </button>
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(3, 1fr)",
                    gap: 10,
                  }}
                >
                  <FormField
                    label="Subpasta treino"
                    value={testForm.train_dir}
                    onChange={(v) => setTestForm((f) => ({ ...f, train_dir: v }))}
                  />
                  <FormField
                    label="Subpasta validação"
                    value={testForm.val_dir}
                    onChange={(v) => setTestForm((f) => ({ ...f, val_dir: v }))}
                  />
                  <FormField
                    label="Subpasta teste"
                    value={testForm.test_dir}
                    onChange={(v) => setTestForm((f) => ({ ...f, test_dir: v }))}
                  />
                </div>
                <FormField
                  label="Rótulo (opcional)"
                  value={testForm.label}
                  onChange={(v) => setTestForm((f) => ({ ...f, label: v }))}
                  placeholder="ex: holdout_2026"
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
                  {testing ? "Avaliando…" : "▶ Rodar teste"}
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
                Nenhum teste executado ainda neste modelo.
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {detail.tests.map((t) => (
                  <TestRow key={t.test_id} test={t} onOpenImage={(src, caption) => setLightbox({ src, caption })} />
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
  mean_accuracy: number | null;
  std_accuracy: number | null;
  mean_f1: number | null;
  std_f1: number | null;
}

/** Per-fold detail section, only rendered when the run.json carries
 *  ``fold_results`` (i.e. it was produced by CrossValidationBlock). */
function CrossValidationDetail({ metrics }: { metrics: Record<string, unknown> }) {
  const folds = metrics["fold_results"];
  const agg = metrics["cv_aggregate"];
  if (!Array.isArray(folds) || folds.length === 0) return null;

  const typed = folds as CVFold[];
  const a = (agg ?? {}) as CVAggregate;

  return (
    <Section
      title={`Cross-validation · ${a.n_folds_ok ?? typed.length}/${a.n_folds ?? typed.length} folds ok${
        (a.n_folds_failed ?? 0) > 0 ? ` · ${a.n_folds_failed} falharam` : ""
      }`}
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
            label="Acurácia média ± std"
            mean={a.mean_accuracy}
            std={a.std_accuracy ?? 0}
            highlight
          />
        )}
        {a.mean_f1 !== null && a.mean_f1 !== undefined && (
          <CVAggregateCard
            label="F1 média ± std"
            mean={a.mean_f1}
            std={a.std_f1 ?? 0}
          />
        )}
      </div>

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
              <th style={cvThStyle}>Fold</th>
              <th style={cvThStyle}>train</th>
              <th style={cvThStyle}>val</th>
              <th style={cvThStyle}>val_loss</th>
              <th style={cvThStyle}>accuracy</th>
              <th style={cvThStyle}>F1</th>
              <th style={cvThStyle}>status</th>
            </tr>
          </thead>
          <tbody>
            {typed.map((f) => {
              const ok = f.status === "success";
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
                      color: ok
                        ? "oklch(0.85 0.16 150)"
                        : "oklch(0.85 0.14 22)",
                    }}
                  >
                    {ok ? "✓" : `× ${f.error || "?"}`}
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
  highlight,
}: {
  label: string;
  mean: number;
  std: number;
  highlight?: boolean;
}) {
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
          ± {std.toFixed(4)}
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

function PathRow({ label, value }: { label: string; value: string }) {
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
        title="Copiar caminho"
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
        copy
      </button>
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

function MetricsGrid({ metrics }: { metrics: Record<string, unknown> }) {
  const entries = Object.entries(metrics);
  if (entries.length === 0) {
    return <div style={{ color: "var(--vf-text-muted)", fontSize: 12 }}>Sem métricas registradas.</div>;
  }
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
        gap: 8,
      }}
    >
      {entries.map(([k, v]) => (
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
            {metricLabel(k)}
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
        </div>
      ))}
    </div>
  );
}

interface TestRowProps {
  test: TestRecord;
  onOpenImage: (src: string, caption: string) => void;
}

function TestRow({ test, onOpenImage }: TestRowProps) {
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
          {new Date(test.timestamp).toLocaleString("pt-BR")}
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
  ];
  const rows = KNOBS.filter(
    (k) => training[k] !== undefined && training[k] !== null,
  ).map((k) => [k, training[k]] as const);

  const tl = getConfigRecord(config, "transfer_learning");
  if (rows.length === 0 && !tl) return null;

  return (
    <Section title="Configuração de treino">
      {rows.map(([k, v]) => (
        <KeyRow key={k} label={k.replace(/_/g, " ")} value={String(v)} />
      ))}
      {tl && (
        <KeyRow
          label="transfer learning"
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
    <Section title="Pipeline aplicado (preprocessing + augmentation)">
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
            // pré-processamento (ordem)
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
            // augmentation & normalize
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
      <ExportStat label="file_size" value={`${sizeMb} MB`} />
      {val && (
        <ExportStat
          label="max_diff"
          value={
            typeof val.max_diff === "number" ? val.max_diff.toExponential(3) : "—"
          }
          accent={val.within_tolerance ? "oklch(0.85 0.16 150)" : "oklch(0.85 0.14 22)"}
        />
      )}
      {bench && (
        <>
          <ExportStat
            label="onnx latency μ"
            value={
              typeof bench.mean_ms === "number"
                ? `${bench.mean_ms.toFixed(2)} ms`
                : "—"
            }
          />
          <ExportStat
            label="onnx p95"
            value={
              typeof bench.p95_ms === "number" ? `${bench.p95_ms.toFixed(2)} ms` : "—"
            }
          />
          <ExportStat
            label="torch latency μ"
            value={
              typeof bench.torch_mean_ms === "number"
                ? `${bench.torch_mean_ms.toFixed(2)} ms`
                : "—"
            }
          />
          <ExportStat
            label="speedup (torch/onnx)"
            value={
              typeof bench.speedup === "number" ? `${bench.speedup.toFixed(2)}×` : "—"
            }
            accent={
              typeof bench.speedup === "number" && bench.speedup >= 1
                ? "oklch(0.85 0.16 150)"
                : undefined
            }
          />
          <ExportStat label="n_runs" value={String(bench.runs ?? "—")} />
        </>
      )}
    </div>
  );
}

function BatchResultPanel({ result }: { result: BatchPredictResponse }) {
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
          label="processadas"
          value={String(result.total_processed)}
          accent="oklch(0.85 0.16 150)"
        />
        <ExportStat
          label="falharam"
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
          csv
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
          title="Copiar caminho"
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
          copy
        </button>
      </div>
      {failedHead.length > 0 && (
        <details style={{ fontSize: 11, color: "var(--vf-text-muted)" }}>
          <summary style={{ cursor: "pointer", color: "oklch(0.85 0.14 22)" }}>
            {result.failed_count} arquivos falharam (clique para ver até 5)
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
              …+{more} mais
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
