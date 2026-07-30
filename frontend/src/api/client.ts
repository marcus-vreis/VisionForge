import type { TaskDescriptor } from "../lib/custom-tasks";
import type { JsonSchema } from "../types/schema";
import type {
  MetricCI,
  QueueSnapshot,
  RunResponse,
  RunResult,
  RunStatus,
  RunSummary,
} from "../types/run";

const BASE = "/api";

export interface FastApiValidationError {
  loc: (string | number)[];
  msg: string;
  type?: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, init);
  } catch (e) {
    throw new ApiError(
      0,
      "Não foi possível conectar ao servidor. Verifique se o backend está rodando.",
      e instanceof Error ? e.message : String(e),
    );
  }

  if (!res.ok) {
    let body: unknown = null;
    try {
      body = await res.json();
    } catch {
      // not JSON
    }

    // FastAPI validation: detail is an array of {loc, msg, type}
    if (
      body &&
      typeof body === "object" &&
      "detail" in body &&
      Array.isArray((body as { detail: unknown }).detail)
    ) {
      throw new ApiError(
        res.status,
        "Erros de validação no formulário.",
        undefined,
        (body as { detail: FastApiValidationError[] }).detail,
      );
    }

    // FastAPI HTTPException: detail is a string
    if (
      body &&
      typeof body === "object" &&
      "detail" in body &&
      typeof (body as { detail: unknown }).detail === "string"
    ) {
      throw new ApiError(
        res.status,
        (body as { detail: string }).detail,
      );
    }

    throw new ApiError(res.status, res.statusText || `HTTP ${res.status}`);
  }
  return res.json();
}

export class ApiError extends Error {
  status: number;
  cause?: string;
  validationErrors?: FastApiValidationError[];

  constructor(
    status: number,
    message: string,
    cause?: string,
    validationErrors?: FastApiValidationError[],
  ) {
    super(message);
    this.status = status;
    this.cause = cause;
    this.validationErrors = validationErrors;
  }
}

export async function fetchSchema(): Promise<JsonSchema> {
  return request<JsonSchema>("/schema");
}

export async function runExperiment(
  config: Record<string, unknown>,
): Promise<RunResponse> {
  return request<RunResponse>("/experiment/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

/** Start an object-detection run. Detection shares the /experiment/{status,
 *  events,result} endpoints, so only the submit endpoint differs. */
export async function runDetection(
  config: Record<string, unknown>,
): Promise<RunResponse> {
  return request<RunResponse>("/detection/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

export async function fetchDetectionSchema(): Promise<JsonSchema> {
  return request<JsonSchema>("/detection/schema");
}

/** Fetch a standalone task's live config schema (drives YAML import validation
 *  in the ExperimentHeader — ADR-059). */
export async function fetchTaskSchema(task: string): Promise<JsonSchema> {
  return request<JsonSchema>(`/${task}/schema`);
}

/** Start an image-regression run. Like detection, regression shares the
 *  /experiment/{status,events,result} endpoints; only the submit URL differs. */
export async function runRegression(
  config: Record<string, unknown>,
): Promise<RunResponse> {
  return request<RunResponse>("/regression/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

/** Start a semantic-segmentation run. Like detection/regression, segmentation
 *  shares the /experiment/{status,events,result} endpoints; only the submit URL
 *  differs. */
export async function runSegmentation(
  config: Record<string, unknown>,
): Promise<RunResponse> {
  return request<RunResponse>("/segmentation/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

/** Start an anomaly-detection run. Shares the /experiment/{status,events,result}
 *  endpoints; only the submit URL differs. */
export async function runAnomaly(
  config: Record<string, unknown>,
): Promise<RunResponse> {
  return request<RunResponse>("/anomaly/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

/** Start a grid/random hyperparameter sweep for a standalone task (ADR-045).
 *  Comparing architectures is the sweep's "arquiteturas" preset — a one-axis
 *  grid over model.name (ADR-059 folded the separate comparison card). */
export async function runSweep(
  task: string,
  payload: Record<string, unknown>,
): Promise<RunResponse> {
  return request<RunResponse>(`/${task}/sweep`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/** K-fold cross-validation over a standalone task's train split (ADR-050:
 *  regression rows / segmentation image-mask pairs). */
export async function runTaskCv(
  task: string,
  payload: Record<string, unknown>,
): Promise<RunResponse> {
  return request<RunResponse>(`/${task}/cv`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/** Train the same config N times under different seeds (ADR-056) and get the
 *  mean/std/95% CI aggregate report. */
export async function runReplicates(
  task: string,
  payload: Record<string, unknown>,
): Promise<RunResponse> {
  return request<RunResponse>(`/${task}/replicates`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/** Every renderable task: the five built-ins + registered custom tasks
 *  (ADR-058). The backend rescans user_tasks/ per call, so a freshly dropped
 *  file appears on reload without restarting the server. */
export async function fetchTasks(): Promise<{ tasks: TaskDescriptor[] }> {
  return request<{ tasks: TaskDescriptor[] }>("/tasks");
}

/** A custom task's Config schema — drives the generic form. */
export async function fetchCustomSchema(key: string): Promise<JsonSchema> {
  return request<JsonSchema>(`/custom/${key}/schema`);
}

/** Start a custom-task run. Shares /experiment/{status,events,result}. */
export async function runCustomTask(
  key: string,
  config: Record<string, unknown>,
): Promise<RunResponse> {
  return request<RunResponse>(`/custom/${key}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
}

/** Grid/random/Optuna sweep over a custom task's own config fields. */
export async function runCustomSweep(
  key: string,
  payload: Record<string, unknown>,
): Promise<RunResponse> {
  return request<RunResponse>(`/custom/${key}/sweep`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/** Multi-seed replicates for a custom task (ADR-056). */
export async function runCustomReplicates(
  key: string,
  payload: Record<string, unknown>,
): Promise<RunResponse> {
  return request<RunResponse>(`/custom/${key}/replicates`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function fetchStatus(): Promise<RunStatus> {
  return request<RunStatus>("/experiment/status");
}

/** The active run plus everything waiting behind it (ADR-075). */
export async function fetchQueue(): Promise<QueueSnapshot> {
  return request<QueueSnapshot>("/queue");
}

/** Drop a submission that has not started. 404 once it is running. */
export async function cancelQueuedRun(
  runId: string,
): Promise<{ run_id: string; status: string }> {
  return request(`/queue/${encodeURIComponent(runId)}`, { method: "DELETE" });
}

export async function fetchResult(runId: string): Promise<RunResult> {
  return request<RunResult>(`/experiment/result/${runId}`);
}

export async function fetchRuns(): Promise<RunSummary[]> {
  return request<RunSummary[]>("/runs");
}

export interface DatasetDetectResponse {
  base_dir: string;
  detected: boolean;
  train_dir: string | null;
  val_dir: string | null;
  test_dir: string | null;
  candidates: string[];
  message: string;
}

export async function detectDatasetSplits(
  baseDir: string,
): Promise<DatasetDetectResponse> {
  return request<DatasetDetectResponse>("/dataset/detect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ base_dir: baseDir }),
  });
}

export interface DatasetPickResponse {
  path: string;
  cancelled: boolean;
  message: string | null;
}

export async function pickDatasetFolder(): Promise<DatasetPickResponse> {
  return request<DatasetPickResponse>("/dataset/pick", { method: "POST" });
}

export interface DatasetDownloadRequest {
  provider: "torchvision" | "roboflow" | "kaggle" | "huggingface";
  dataset: string;
  out_dir: string;
  splits?: string[];
  limit?: number | null;
  api_key?: string | null;
  token?: string | null;
  version?: number | null;
  dataset_format?: string | null;
}

export interface DatasetDownloadResponse {
  provider: string;
  dataset: string;
  out_dir: string;
  total_images: number;
  splits: Record<string, number>;
  classes: string[];
}

/** One-shot download of a dataset to a local folder (ADR-055). */
export async function datasetDownload(
  req: DatasetDownloadRequest,
): Promise<DatasetDownloadResponse> {
  return request<DatasetDownloadResponse>("/dataset/download", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

/** Open a native file picker filtered to .yaml/.yml — used to select an
 *  existing Ultralytics data.yaml for a detection run. */
export async function pickDetectionYaml(): Promise<DatasetPickResponse> {
  return request<DatasetPickResponse>("/detection/dataset/pick_yaml", {
    method: "POST",
  });
}

export interface CheckpointPickResponse {
  path: string;
  cancelled: boolean;
  message: string | null;
}

export async function pickCheckpointFile(): Promise<CheckpointPickResponse> {
  return request<CheckpointPickResponse>("/checkpoint/pick", { method: "POST" });
}

export interface ExportOnnxRequestPayload {
  output_onnx?: string | null;
  opset_version?: number;
  dynamic_axes?: boolean;
  validate?: boolean;
  benchmark?: boolean;
  benchmark_runs?: number;
}

export interface ExportOnnxResponse {
  output_onnx: string;
  file_size_bytes: number;
  validation: {
    max_diff?: number;
    within_tolerance?: boolean;
    tolerance?: number;
  } | null;
  benchmark: {
    mean_ms?: number;
    p50_ms?: number;
    p95_ms?: number;
    torch_mean_ms?: number;
    speedup?: number;
    runs?: number;
  } | null;
}

export async function exportRunToOnnx(
  runId: string,
  payload: ExportOnnxRequestPayload,
): Promise<ExportOnnxResponse> {
  return request<ExportOnnxResponse>(
    `/runs/${encodeURIComponent(runId)}/export_onnx`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
}

export interface BatchPredictRequestPayload {
  input_dir: string;
  output_csv?: string | null;
  recursive?: boolean;
  class_names?: string[] | null;
}

export interface BatchPredictResponse {
  output_csv: string;
  total_processed: number;
  failed_count: number;
  failed_files: string[];
}

export async function batchPredictRun(
  runId: string,
  payload: BatchPredictRequestPayload,
): Promise<BatchPredictResponse> {
  return request<BatchPredictResponse>(
    `/runs/${encodeURIComponent(runId)}/batch_predict`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
}

export interface GradCamRequestPayload {
  input_dir: string;
  num_samples?: number;
  target_class?: number | null;
  alpha?: number;
  recursive?: boolean;
}

export interface GradCamItem {
  source: string;
  overlay: string;
  /** argmax class (classification/segmentation); null for regression. */
  predicted_class: number | null;
  /** human-readable label (regression values / segmentation target); may be null. */
  prediction?: string | null;
}

export interface GradCamResponse {
  run_id: string;
  count: number;
  target_layer: string;
  items: GradCamItem[];
}

/** Generate Grad-CAM overlays for a trained classification run's checkpoint. */
export async function gradcamRun(
  runId: string,
  payload: GradCamRequestPayload,
): Promise<GradCamResponse> {
  return request<GradCamResponse>(
    `/runs/${encodeURIComponent(runId)}/gradcam`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
}

export interface SplitStats {
  total_images: number;
  classes: Record<string, number>;
  missing: boolean;
}

export interface DatasetStatsResponse {
  base_dir: string;
  splits: Record<string, SplitStats>;
  class_names: string[];
  imbalanced: boolean;
  message: string | null;
}

export async function fetchDatasetStats(
  baseDir: string,
  splits?: { train_dir?: string; val_dir?: string; test_dir?: string },
): Promise<DatasetStatsResponse> {
  return request<DatasetStatsResponse>("/dataset/stats", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ base_dir: baseDir, ...(splits ?? {}) }),
  });
}

export interface DetectionSplitStats {
  total_images: number;
  total_annotations: number;
  class_counts: Record<string, number>;
  unlabeled_images: number;
  missing: boolean;
}

export interface DetectionDatasetStatsResponse {
  base_dir: string;
  splits: Record<string, DetectionSplitStats>;
  class_names: string[];
  imbalanced: boolean;
  message: string | null;
}

export async function fetchDetectionDatasetStats(
  baseDir: string,
): Promise<DetectionDatasetStatsResponse> {
  return request<DetectionDatasetStatsResponse>("/detection/dataset/stats", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ base_dir: baseDir }),
  });
}

export interface SegmentationSplitStats {
  images: number;
  masks: number;
  paired: number;
  unpaired_images: number;
  unpaired_masks: number;
  missing: boolean;
}

export interface SegmentationDatasetStatsResponse {
  base_dir: string;
  splits: Record<string, SegmentationSplitStats>;
  mask_class_ids: number[];
  message: string | null;
}

/** Per-split image↔mask pairing stats + sampled class ids (ADR-059 brick E). */
export async function fetchSegmentationDatasetStats(body: {
  base_dir: string;
  images_subdir: string;
  masks_subdir: string;
  train_dir: string;
  val_dir: string;
  test_dir: string;
}): Promise<SegmentationDatasetStatsResponse> {
  return request<SegmentationDatasetStatsResponse>(
    "/segmentation/dataset/stats",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
  );
}

export interface AnomalyDatasetStatsResponse {
  base_dir: string;
  train_normal: number;
  test_normal: number;
  test_anomalous: Record<string, number>;
  missing_train: boolean;
  missing_test: boolean;
  message: string | null;
}

/** Normal-vs-defect counts for an MVTec-style dataset (ADR-059 brick E). */
export async function fetchAnomalyDatasetStats(body: {
  base_dir: string;
  train_dir: string;
  test_dir: string;
  normal_dir: string;
}): Promise<AnomalyDatasetStatsResponse> {
  return request<AnomalyDatasetStatsResponse>("/anomaly/dataset/stats", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export interface RegressionTargetStats {
  count: number;
  mean: number | null;
  min: number | null;
  max: number | null;
}

export interface RegressionSplitStats {
  rows: number;
  missing_columns: string[];
  missing_images: number;
  checked_images: number;
  targets: Record<string, RegressionTargetStats>;
  missing: boolean;
}

export interface RegressionDatasetStatsResponse {
  base_dir: string;
  splits: Record<string, RegressionSplitStats>;
  message: string | null;
}

/** Manifest rows, column checks + target distributions (ADR-059 brick E). */
export async function fetchRegressionDatasetStats(body: {
  base_dir: string;
  images_dir: string;
  train_csv: string;
  val_csv: string;
  test_csv: string;
  image_column: string;
  target_columns: string[];
}): Promise<RegressionDatasetStatsResponse> {
  return request<RegressionDatasetStatsResponse>("/regression/dataset/stats", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export interface DatasetSamplesResponse {
  base_dir: string;
  split: string;
  samples: Record<string, string[]>;
  message: string | null;
}

export async function fetchDatasetSamples(
  baseDir: string,
  split: "train" | "val" | "test" = "train",
  perClass = 4,
): Promise<DatasetSamplesResponse> {
  return request<DatasetSamplesResponse>("/dataset/samples", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      base_dir: baseDir,
      split,
      per_class: perClass,
    }),
  });
}

/** Build a URL that fetches a local image file from the dataset for previewing. */
export function datasetFileUrl(absolutePath: string): string {
  return `/api/dataset/file?path=${encodeURIComponent(absolutePath)}`;
}

export interface GPUInfo {
  index: number;
  name: string;
  total_memory_mb: number;
  compute_capability: string | null;
}

export interface DeviceInfoResponse {
  cuda_available: boolean;
  cuda_version: string | null;
  cpu_name: string;
  gpus: GPUInfo[];
}

export async function fetchDeviceInfo(): Promise<DeviceInfoResponse> {
  return request<DeviceInfoResponse>("/device/info");
}

export interface PreprocessPreviewStep {
  kind: string;
  artifact: string;
  params: Record<string, unknown>;
}

export interface PreprocessPreviewResponse {
  original: string;
  steps: PreprocessPreviewStep[];
  final: string;
  source_image: string;
  available_kinds: string[];
  message: string | null;
}

export async function previewPreprocess(
  baseDir: string,
  steps: Array<{ kind: string } & Record<string, unknown>>,
  options?: { split?: "train" | "val" | "test"; className?: string },
): Promise<PreprocessPreviewResponse> {
  return request<PreprocessPreviewResponse>("/dataset/preview_preprocess", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      base_dir: baseDir,
      split: options?.split ?? "train",
      class_name: options?.className ?? null,
      steps,
    }),
  });
}

export interface AugmentPreviewResponse {
  original: string;
  variants: string[];
  source_image: string;
  active: string[];
  message: string | null;
}

/** Render N randomly-augmented variants of a sample image so the user can see
 *  the train-time augmentation effect (flip/rotation/jitter) before training. */
export async function previewAugment(
  baseDir: string,
  transforms: Record<string, unknown>,
  options?: {
    split?: "train" | "val" | "test";
    className?: string;
    numVariants?: number;
  },
): Promise<AugmentPreviewResponse> {
  return request<AugmentPreviewResponse>("/dataset/preview_augment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      base_dir: baseDir,
      split: options?.split ?? "train",
      class_name: options?.className ?? null,
      transforms,
      num_variants: options?.numVariants ?? 4,
    }),
  });
}


export interface SystemInfo {
  cpu_count: number;
  suggested_workers: number;
  platform: string;
  version: string;
}

export async function fetchSystemInfo(): Promise<SystemInfo> {
  return request<SystemInfo>("/system/info");
}

export interface RunDetail {
  run_id: string;
  experiment_name: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  device_used: string | null;
  environment?: Record<string, string>;
  run_dir: string;
  config: Record<string, unknown>;
  metrics: Record<string, unknown>;
  /** Bootstrap interval per test metric (ADR-074), keyed by bare metric name. */
  metric_cis?: Record<string, MetricCI>;
  history: Array<{
    epoch: number;
    train_loss: number;
    train_accuracy: number;
    val_loss: number;
    val_accuracy: number;
  }>;
  artifacts: {
    model?: string;
    graphics?: string[];
    report?: string | null;
  };
  tests: TestRecord[];
}

export interface TestRecord {
  test_id: string;
  label: string;
  base_dir: string;
  timestamp: string;
  metrics: Record<string, number | null>;
  artifacts: Record<string, string>;
}

export async function fetchRunDetail(runId: string): Promise<RunDetail> {
  return request<RunDetail>(`/runs/${encodeURIComponent(runId)}`);
}

/** Permanently delete a run directory. The backend refuses to delete the
 *  currently running run (409) and rejects anything outside _MODELS_DIR (400). */
export async function deleteRun(runId: string): Promise<{ run_id: string; status: string }> {
  return request(`/runs/${encodeURIComponent(runId)}`, { method: "DELETE" });
}

export interface RunTestRequestPayload {
  base_dir: string;
  train_dir?: string;
  val_dir?: string;
  test_dir?: string;
  label?: string;
}

export async function testRunOnDataset(
  runId: string,
  payload: RunTestRequestPayload,
): Promise<TestRecord> {
  return request<TestRecord>(`/runs/${encodeURIComponent(runId)}/test`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/** Download the model card markdown for a run. Triggers a browser download. */
export async function downloadRunMarkdown(runId: string): Promise<void> {
  const res = await fetch(`${BASE}/runs/${encodeURIComponent(runId)}/export_md`);
  if (!res.ok) {
    throw new ApiError(res.status, `HTTP ${res.status} ao gerar markdown.`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  // Use the server-supplied filename when present, otherwise fallback.
  const cd = res.headers.get("content-disposition") ?? "";
  const match = cd.match(/filename="([^"]+)"/);
  a.download = match ? match[1] : `${runId}.md`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function artifactUrl(path: string): string {
  return `${BASE}/artifacts/${path}`;
}

/** Whether each dataset provider has a stored key, and its masked form.
 *
 * The real value never reaches the browser: the GUI only needs to show that a
 * key exists and which one, and the download runs server-side.
 */
export interface CredentialEntry {
  saved: boolean;
  masked: string;
}

export interface CredentialsResponse {
  providers: Record<string, CredentialEntry>;
  config_dir: string;
}

export async function fetchCredentials(): Promise<CredentialsResponse> {
  return request<CredentialsResponse>("/credentials");
}

export async function saveCredential(
  provider: string,
  value: string,
): Promise<CredentialsResponse> {
  return request<CredentialsResponse>("/credentials", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, value }),
  });
}

export async function forgetCredential(
  provider: string,
): Promise<CredentialsResponse> {
  return request<CredentialsResponse>(`/credentials/${provider}`, {
    method: "DELETE",
  });
}

/** Outcome of hiding, unhiding or deleting a researcher-defined task. */
export interface CustomTaskActionResponse {
  key: string;
  action: "hidden" | "unhidden" | "deleted";
  detail: string;
}

export async function hideCustomTask(
  key: string,
): Promise<CustomTaskActionResponse> {
  return request<CustomTaskActionResponse>(`/custom/${key}/hide`, {
    method: "POST",
  });
}

/** Delete a custom task's source file. `confirm` must repeat the key — there
 *  is no undo, so the confirmation is deliberately more work than a click. */
export async function deleteCustomTask(
  key: string,
  confirm: string,
): Promise<CustomTaskActionResponse> {
  return request<CustomTaskActionResponse>(
    `/custom/${key}?confirm=${encodeURIComponent(confirm)}`,
    { method: "DELETE" },
  );
}
