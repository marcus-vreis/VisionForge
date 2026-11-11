import { en } from "../i18n/en";
import {
  initialLang,
  langHeaders,
  readStoredLang,
  type Lang,
} from "../i18n/lang";
import { pt, type Dict } from "../i18n/pt";
import type { TaskDescriptor } from "../lib/custom-tasks";
import {
  explorerBody,
  type ExplorerForm,
  type ExplorerSummary,
  type ScanJob,
} from "../lib/dataset-explorer";
import type {
  IncorporateJob,
  IncorporatePlan,
  IncorporatePreview,
  IncorporateResult,
  PlanBody,
} from "../lib/dataset-incorporate";
import type {
  MappingBody,
  MergePlan,
  MergePreview,
  MergeResult,
} from "../lib/dataset-merge";
import type {
  CocoExportPreview,
  CocoExportResult,
  CocoImportPlan,
  CocoImportPreview,
  CocoImportResult,
} from "../lib/dataset-coco";
import type {
  ResplitBody,
  ResplitPlan,
  ResplitPreview,
  ResplitResult,
} from "../lib/dataset-resplit";
import {
  activeProfileSlug,
  profileHeaders,
  type ProfileInfo,
  type Pronoun,
} from "../lib/profile";
import type { JsonSchema } from "../types/schema";
import type {
  DatasetInfo,
  MetricCI,
  QueueSnapshot,
  RunGroup,
  RunResponse,
  RunResult,
  RunStatus,
  RunSummary,
} from "../types/run";

const BASE = "/api";

export interface DatasetAuditRequest {
  base_dir: string;
  splits?: Record<string, string> | null;
  group_manifest?: string | null;
  strict_groups?: boolean;
}

export interface DatasetAuditReport {
  ok: boolean;
  root: string;
  hash_algorithm: "sha256";
  visual_similarity_checked: false;
  image_counts: Record<string, number>;
  duplicate_groups: { sha256: string; splits: string[]; paths: string[] }[];
  missing_splits: string[];
  empty_splits: string[];
  unreadable_count: number;
  unreadable: { path: string; reason: string }[];
  group_manifest: string | null;
  strict_groups: boolean;
  group_overlaps: { group: string; splits: string[]; paths: string[] }[];
}

/** Explicitly audit image splits for exact duplicates and group leakage. */
export async function auditDataset(body: DatasetAuditRequest): Promise<DatasetAuditReport> {
  return request<DatasetAuditReport>("/dataset/audit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** Download the run's evidence ZIP with the active profile and language. */
export async function downloadRunBundle(runId: string): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/runs/${encodeURIComponent(runId)}/bundle`, withContext());
  } catch (error) {
    throw new ApiError(0, errorTexts().cannotConnect,
      error instanceof Error ? error.message : String(error));
  }
  if (!res.ok) {
    let detail: unknown;
    try { detail = (await res.json() as { detail?: unknown }).detail; } catch { /* Non-JSON error. */ }
    throw new ApiError(res.status, typeof detail === "string" ? detail : `HTTP ${res.status}`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  const disposition = res.headers.get("content-disposition") ?? "";
  anchor.download = disposition.match(/filename="([^"]+)"/)?.[1] ?? `${runId}.zip`;
  try {
    document.body.appendChild(anchor);
    anchor.click();
  } finally {
    anchor.remove();
    URL.revokeObjectURL(url);
  }
}

/**
 * The language on screen, resolved the way the provider seeds it.
 *
 * This module is not a component, so it cannot call useT(), and threading the
 * language through every request would touch each of its ~50 callers.
 * setLang() stores the choice before anything can fail, so this follows the
 * header toggle on the next call.
 */
function currentLang(): Lang {
  return initialLang(
    readStoredLang(),
    typeof navigator === "undefined" ? undefined : navigator.language,
  );
}

/** The texts of the errors raised here, built when they are thrown. */
function errorTexts(): Dict["errors"] {
  return (currentLang() === "pt" ? pt : en).errors;
}

export interface FastApiValidationError {
  loc: (string | number)[];
  msg: string;
  type?: string;
}

/**
 * `init` with the profile header added (ADR-114).
 *
 * Every call carries the chosen profile, so which folder a request reads or
 * writes is never a decision a single endpoint has to remember. The default
 * profile sends nothing: no header already means the default.
 */
export function withProfile(
  init?: RequestInit,
  slug: string = activeProfileSlug(),
): RequestInit {
  const extra = profileHeaders(slug);
  if (Object.keys(extra).length === 0) return init ?? {};
  const headers = new Headers(init?.headers);
  for (const [name, value] of Object.entries(extra)) headers.set(name, value);
  return { ...init, headers };
}

/**
 * `init` with the interface language added (ADR-116).
 *
 * The server words the messages it composes (a dataset scan, a health warning,
 * a refused request) in this language. Without the header it writes Portuguese.
 */
export function withLang(
  init?: RequestInit,
  lang: Lang = currentLang(),
): RequestInit {
  const headers = new Headers(init?.headers);
  for (const [name, value] of Object.entries(langHeaders(lang))) {
    headers.set(name, value);
  }
  return { ...init, headers };
}

/**
 * Every call carries the chosen profile and the language on screen. `profile`
 * names another one, for a setting edited in a profile's own row.
 */
function withContext(init?: RequestInit, profile?: string): RequestInit {
  return withLang(withProfile(init, profile));
}

async function request<T>(path: string, init?: RequestInit, profile?: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, withContext(init, profile));
  } catch (e) {
    throw new ApiError(
      0,
      errorTexts().cannotConnect,
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
        errorTexts().validation,
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
        undefined,
        undefined,
        body,
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
  /** The parsed error body, for the few endpoints whose error carries data next
   *  to `detail` (a 409 that names the folder already there). */
  body?: unknown;

  constructor(
    status: number,
    message: string,
    cause?: string,
    validationErrors?: FastApiValidationError[],
    body?: unknown,
  ) {
    super(message);
    this.status = status;
    this.cause = cause;
    this.validationErrors = validationErrors;
    this.body = body;
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

/** Train N architectures of a standalone task on the same config and rank
 *  them (ADR-044). */
export async function runCompare(
  task: string,
  payload: Record<string, unknown>,
): Promise<RunResponse> {
  return request<RunResponse>(`/${task}/compare`, {
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

/** A run's status; with no `runId`, the job started last. A local and a Colab
 *  job run side by side (ADR-163, phase 4), so a page following one names it. */
export async function fetchStatus(runId?: string | null): Promise<RunStatus> {
  const query = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  return request<RunStatus>(`/experiment/status${query}`);
}

/** The active run plus everything waiting behind it (ADR-075). */
export async function fetchQueue(): Promise<QueueSnapshot> {
  return request<QueueSnapshot>("/queue");
}

/** Stop a submission, whether it is waiting or already training.
 *
 * A waiting job is removed at once. A running one is only *asked* to stop: the
 * 200 means the request was delivered, and the trainer ends the run at the top
 * of its next epoch, keeping the best checkpoint and the history so far
 * (ADR-088, ADR-094). It is 404 when no job carries that id — already finished,
 * or never submitted. Not every kind of run listens; lib/run-control.ts says
 * which. */
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

/** Every profile on the server, the default one first (ADR-114). */
export async function fetchProfiles(): Promise<ProfileInfo[]> {
  const body = await request<{ profiles: ProfileInfo[] }>("/profiles");
  return body.profiles;
}

/** Create a profile folder from a display name. 409 when its folder exists and
 *  422 when nothing usable is left of the name. */
export async function createProfile(name: string, pronoun?: Pronoun): Promise<ProfileInfo> {
  return request<ProfileInfo>("/profiles", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, pronoun }),
  });
}

/** Novo nome e/ou pronome de um perfil, o padrão inclusive (ADR-126). */
export async function updateProfile(
  slug: string,
  change: {
    name?: string; pronoun?: Pronoun;
    settings?: { mlflow?: { enabled: boolean }; wandb?: Partial<WandbSettings> };
  },
): Promise<ProfileInfo> {
  return request<ProfileInfo>(`/profiles/${encodeURIComponent(slug)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(change),
  });
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

/** The synthetic sample dataset of the "Primeiro treino" guide (ADR-115, ADR-117). */
export interface SampleDataset {
  task: string;
  /** Absolute path of the folder on the server's disk. */
  path: string;
  /** What the layout declares: class folders, the names in `classes.txt`, the mask
   *  pixel values or the subfolders of an anomaly dataset's `test/`. */
  classes: string[];
  /** The CSV target columns of a regression dataset. */
  targets: string[];
  /** Items per split, in the unit the task reads (images, CSV rows, image and mask
   *  pairs). Empty when the folder already existed. */
  counts: Record<string, number>;
  /** The folder was already there and was reused, not created now. */
  existed: boolean;
}

/** Create `datasets/exemplo-<task>/` on the server. A 409 means the folder is
 *  already there: the call then resolves with `existed: true` and its path, so the
 *  guide can use it. Any other failure rejects. */
export async function createSampleDataset(
  task = "classification",
): Promise<SampleDataset> {
  try {
    const made = await request<Omit<SampleDataset, "existed">>(
      "/sample-dataset",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task }),
      },
    );
    return { ...made, existed: false };
  } catch (e) {
    const path =
      e instanceof ApiError && e.status === 409
        ? (e.body as { path?: unknown } | null | undefined)?.path
        : undefined;
    if (typeof path === "string" && path !== "") {
      return { task, path, classes: [], targets: [], counts: {}, existed: true };
    }
    throw e;
  }
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
  /** class-name form of predicted_class, when the run's class list is known. */
  predicted_label?: string | null;
  /** from the image's parent folder; null when the folder carries no label. */
  true_class?: string | null;
  correct?: boolean | null;
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
  /** Which YOLO layout matched, relative to base_dir. Both
   *  `images/<split>` and `<split>/images` are valid, and a dataset that
   *  half-follows one resolves in a way the researcher did not intend. */
  images_dir?: string | null;
  labels_dir?: string | null;
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

/** Per-class box crops, as PNG data URIs.
 *
 * Detection cannot hand back file paths the way classification does: a class is
 * a box inside an image that may contain several, so a path would show a
 * picture and leave the reader guessing which part of it the label meant.
 */
export interface DetectionDatasetSamplesResponse {
  base_dir: string;
  split: string;
  crops: Record<string, string[]>;
  message: string | null;
}

export async function fetchDetectionDatasetSamples(
  baseDir: string,
): Promise<DetectionDatasetSamplesResponse> {
  return request<DetectionDatasetSamplesResponse>("/detection/dataset/samples", {
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

/** Per-split image↔mask pairing stats + sampled class ids (ADR-059). */
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

/** Normal-vs-defect counts for an MVTec-style dataset (ADR-059). */
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

/** Manifest rows, column checks + target distributions (ADR-059). */
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
  /** The optional `detection` extra (ultralytics, AGPL-3.0) is importable (ADR-195). */
  ultralytics_installed: boolean;
  /** The line that installs that extra for this kind of install. */
  detection_install_command: string;
}

/** What was measured for an architecture's family at Adam 1e-3 (ADR-099/100):
 *  the one model that was run, the accuracy it reached on 4 classes, and whether
 *  it collapsed to a single class or merely failed to learn. */
export interface CollapseEvidence {
  measured_on: string;
  accuracy: number;
  outcome: "collapse" | "fails_to_learn";
  /** What `measured_on` reached at the suggested setting on the same data. Null
   *  (or absent, from an older server) where that run was never made on this
   *  setup: the note then only suggests the setting and claims nothing about it. */
  recovered_accuracy?: number | null;
}

export interface ModelDefaults {
  architecture: string;
  optimizer: string;
  learning_rate: number;
  image_size: number | null;
  dataset_median_side: number | null;
  collapse_prone: boolean;
  /** The measurement behind `collapse_prone`. Null when the family was never
   *  run, and absent from a server that predates the field; either way the
   *  interface must not quote an accuracy. */
  collapse_evidence?: CollapseEvidence | null;
  /** The server's own sentence, in Portuguese. Not shown: ModelAdvice words the
   *  same cases from the fields above, in the language of the interface. */
  note: string | null;
}

export interface LearningRatePlanRequest {
  block: string;
  model_name: string;
  optimizer: string;
  learning_rate: number;
  hyperparameters: Record<string, unknown[]>;
  lr_mode: "absolute" | "suggested_multiplier";
  model_names: string[];
  lr_policy: "fixed" | "suggested";
}

/** One combination of a job and the rate it will train at (ADR-158). */
export interface TrialLearningRate {
  architecture: string;
  optimizer: string;
  learning_rate: number;
  /** The factor of the suggested rate; null when the rate is not a factor's. */
  multiplier: number | null;
  collapse_prone: boolean;
  collapse_evidence: CollapseEvidence | null;
}

export async function fetchLearningRatePlan(
  body: LearningRatePlanRequest,
): Promise<{ trials: TrialLearningRate[] }> {
  return request<{ trials: TrialLearningRate[] }>("/model/learning-rate-plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** Suggested settings for a given architecture (ADR-099/100). Offered to the
 *  form, never applied behind the researcher. */
export async function fetchModelDefaults(
  architecture: string,
  baseDir?: string,
  pretrained = true,
): Promise<ModelDefaults> {
  return request<ModelDefaults>("/model/defaults", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ architecture, base_dir: baseDir ?? null, pretrained }),
  });
}

export async function fetchSystemInfo(): Promise<SystemInfo> {
  return request<SystemInfo>("/system/info");
}

/** A profile's Weights & Biases destination (ADR-172); `entity` empty = the key's user. */
export interface WandbSettings {
  enabled: boolean;
  project: string;
  entity: string;
}

/** What the History knows about one run's W&B copy. `url` is always https. */
export interface WandbExportStatus {
  state: "running" | "finished" | "disabled" | "failed";
  status: "RUNNING" | "FINISHED" | "KILLED" | "FAILED";
  error?: string | null;
  url?: string | null;
  project?: string | null;
  entity?: string | null;
}

/** Whether `profile` has a saved W&B key; the key itself never comes back. */
export async function fetchWandbToken(profile: string): Promise<HubTokenStatus> {
  return request<HubTokenStatus>("/wandb/token", undefined, profile);
}

/** Save the W&B API key of `profile` on the server. */
export async function saveWandbToken(profile: string, token: string): Promise<HubTokenStatus> {
  return request<HubTokenStatus>("/wandb/token", {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  }, profile);
}

/** Forget the W&B API key of `profile`. */
export async function deleteWandbToken(profile: string): Promise<HubTokenStatus> {
  return request<HubTokenStatus>("/wandb/token", { method: "DELETE" }, profile);
}

/** Send a finished run to its W&B run again, with the active profile's key. */
export async function retryWandbExport(runId: string): Promise<WandbExportStatus> {
  return request<WandbExportStatus>(`/wandb/runs/${encodeURIComponent(runId)}/retry`, { method: "POST" });
}

/** The settings that pick a branch in a run's code (ADR-173); a missing one is undecided.
 *  Each task reads its own (`GET /api/code-tour` lists them); `model` is a model
 *  name the server turns into the setting it decides. */
export interface CodeSettings {
  kind?: "binary" | "multiclass";
  amp?: boolean;
  augment?: boolean;
  model?: string;
  loss?: string;
  method?: "autoencoder" | "patchcore";
  backend?: "torchvision" | "ultralytics";
}

/** A task the "Código" tab has a tour for, and the values of each setting it reads. */
export interface CodeTourTask {
  task: string;
  settings: Record<string, string[]>;
}

/** Lines `first`..`last` (0-based) of a section and what they do. */
export interface CodeNote {
  first: number;
  last: number;
  text: string;
  /** true: this run's path; false: a branch it does not take; null: unconditional or unknown. */
  active: boolean | null;
}

/** One real function of the training, read from the installed package. */
export interface CodeSection {
  id: string;
  name: string;
  file: string;
  first_line: number;
  lines: string[];
  notes: CodeNote[];
}

/** The annotated training code of `task`, with this run's branches marked. */
export async function fetchCodeTour(task: string, settings: CodeSettings): Promise<CodeSection[]> {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(settings)) {
    if (value !== undefined) query.set(key, String(value));
  }
  const suffix = query.size ? `?${query}` : "";
  return request<CodeSection[]>(`/code-tour/${encodeURIComponent(task)}${suffix}`);
}

/** The tasks whose training code the "Código" tab can show. */
export async function fetchCodeTourTasks(): Promise<CodeTourTask[]> {
  return request<CodeTourTask[]>("/code-tour");
}

export interface MLflowTrackingStatus {
  state: string;
  error?: string | null;
  run_id?: string | null;
  status?: string | null;
}

/** Replay the local record into its original MLflow run. */
export async function retryMLflowExport(runId: string): Promise<MLflowTrackingStatus> {
  return request<MLflowTrackingStatus>(`/runs/${encodeURIComponent(runId)}/mlflow/retry`, { method: "POST" });
}

export interface RunDetail {
  tracking?: MLflowTrackingStatus | null;
  wandb?: WandbExportStatus | null;
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
  /** One record per epoch, under the names its trainer writes (lib/compare-curves.ts
   *  lists them per task); a metric a backend does not measure is null. Empty on a
   *  replicate group. */
  history: Array<{ epoch: number; [series: string]: number | null | undefined }>;
  artifacts: {
    model?: string;
    graphics?: string[];
    report?: string | null;
  };
  tests: TestRecord[];
  dataset?: DatasetInfo | null;
  /** See RunSummary: the run stopped short and can be continued (ADR-092/093). */
  resumable?: boolean;
  configured_epochs?: number | null;
  /** A user stop cut this run (ADR-111); same fact as on RunSummary. */
  stopped?: boolean;
  /** A seed of a replicate group names the group (ADR-113). */
  group_id?: string | null;
  /** On the group itself: its seeds, aggregates and, for a replicated
   *  comparison, the variants and paired tests, as the report recorded them. */
  group?: RunGroup | null;
  /** The task family: a built-in name, or `custom:<key>` for a researcher's own
   *  task. Absent from a server older than the comparison by task. */
  task?: string;
  /** The request came from the machine the server runs on, so the file
   *  manager's window opens in front of the person who clicked. Absent on an
   *  older server, which reads as "do not offer the button". */
  can_reveal?: boolean;
  /** Which way each metric and history series improves, by the rule the server
   *  ranks with (a researcher's declared directions included). Absent from a
   *  server that predates it: the page then falls back to the metric's name
   *  (`inferMetricDirection`). */
  metric_directions?: Record<string, "higher" | "lower">;
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

/** Continue a run that stopped before its last epoch (ADR-092/093). The config
 *  comes from the run's own run.json, so the continued run is the same
 *  experiment; the backend answers 409 when there is nothing left to continue. */
export async function resumeRun(runId: string): Promise<RunResponse> {
  return request<RunResponse>(`/runs/${encodeURIComponent(runId)}/resume`, {
    method: "POST",
  });
}

/** Open the run's folder in the file manager of the machine running the server.
 *  The folder is found from the run id alone; the backend answers 403 to a
 *  client that is not that machine and 404 to an unknown run. */
export async function revealRunFolder(
  runId: string,
): Promise<{ run_id: string; run_dir: string }> {
  return request(`/runs/${encodeURIComponent(runId)}/reveal`, { method: "POST" });
}

/** Permanently delete a run directory. The backend refuses to delete the
 *  currently running run (409) and rejects anything outside the profile's models
 *  folder (400); a run of another profile is a 404 (ADR-114). */
export async function deleteRun(runId: string): Promise<{ run_id: string; status: string }> {
  return request(`/runs/${encodeURIComponent(runId)}`, { method: "DELETE" });
}

export interface RunTestRequestPayload {
  /** The single labelled folder to score — or, for regression, the .csv
   *  manifest its data model requires (ADR-080). */
  data_dir: string;
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
  const res = await fetch(
    `${BASE}/runs/${encodeURIComponent(runId)}/export_md`,
    withContext(),
  );
  if (!res.ok) {
    throw new ApiError(res.status, errorTexts().markdownExport(res.status));
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

/** Where this profile's Colab connection stands (ADR-163). The token never
 *  reaches the browser: `account` is the e-mail of the Google account. */
export interface ColabStatus {
  pinned_version: string;
  installed_version: string | null;
  install_state: "idle" | "installing" | "failed";
  install_error: string | null;
  logged_in: boolean;
  account: string | null;
  login_pending: boolean;
}

/** The account's compute units; `hours_left` is null while nothing runs. */
export interface ColabUsage {
  balance: number | null;
  rate_per_hour: number | null;
  active_assignments: number | null;
  hours_left: number | null;
}

export async function fetchColabStatus(): Promise<ColabStatus> {
  return request<ColabStatus>("/colab/status");
}

export async function installColabCli(): Promise<ColabStatus> {
  return request<ColabStatus>("/colab/install", { method: "POST" });
}

export async function startColabLogin(): Promise<{ url: string }> {
  return request<{ url: string }>("/colab/login/start", { method: "POST" });
}

export async function finishColabLogin(code: string): Promise<ColabStatus> {
  return request<ColabStatus>("/colab/login/finish", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code }),
  });
}

export async function cancelColabLogin(): Promise<ColabStatus> {
  return request<ColabStatus>("/colab/login/cancel", { method: "POST" });
}

export async function logoutColab(): Promise<ColabStatus> {
  return request<ColabStatus>("/colab/logout", { method: "POST" });
}

export async function fetchColabUsage(): Promise<ColabUsage> {
  return request<ColabUsage>("/colab/usage");
}

/** Where the VM takes the dataset from (ADR-163, phase 2): a public zip, or a
 *  public dataset it downloads itself. No credential leaves this machine. */
export type ColabDatasetSource =
  | { kind: "url"; url: string }
  | { kind: "provider"; provider: "torchvision" | "huggingface"; dataset: string; limit?: number }
  // The form's own dataset folder, inside Google Drive for desktop (ADR-164).
  | { kind: "drive"; my_drive_root?: string };

export type ColabAccelerator = "T4" | "L4" | "A100" | "H100" | "G4";

/** What a Colab run adds to the classification payload. */
export interface ColabLaunch {
  source: ColabDatasetSource;
  accelerator: ColabAccelerator;
  /** Also save the run to MyDrive/VisionForge/runs/ (ADR-164). */
  drive_export?: boolean;
}

/** Tell a Colab job that the Drive access was granted, so it answers the CLI (ADR-164). */
export async function continueDriveConsent(runId: string): Promise<{ sent: boolean }> {
  return request<{ sent: boolean }>("/colab/drive/continue", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId }),
  });
}

export async function runColabExperiment(body: Record<string, unknown>): Promise<RunResponse> {
  return request<RunResponse>("/colab/experiment/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** A runtime alive on the account; `ours` when VisionForge started it. */
export interface ColabSessionInfo {
  name: string;
  hardware: string;
  ours: boolean;
}

export async function fetchColabSessions(): Promise<ColabSessionInfo[]> {
  return request<ColabSessionInfo[]>("/colab/sessions");
}

export async function stopColabSession(name: string): Promise<ColabSessionInfo[]> {
  return request<ColabSessionInfo[]>("/colab/sessions/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
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

/** What `GET /api/update` answers (ADR-120). */
export interface UpdateStatus {
  current: string;
  latest: string | null;
  available: boolean;
  can_apply: boolean;
  /** Why the button cannot install; worded by the page. */
  blocked: "source" | "no_installer" | "remote" | "busy" | null;
  /** What to type by hand when the button cannot do it. */
  command: string | null;
  last_attempt: {
    target: string;
    ok: boolean;
    returncode: number;
    finished_at: string;
    log: string;
  } | null;
  check_disabled: boolean;
}

/** Whether a newer release exists and whether this page may install it. */
export async function fetchUpdateStatus(): Promise<UpdateStatus> {
  return request<UpdateStatus>("/update");
}

/** Install the announced release; the server restarts right after answering. */
export async function applyUpdate(): Promise<{ status: string; target: string }> {
  return request<{ status: string; target: string }>("/update/apply", { method: "POST" });
}

/** The version a server answers with, or null while it is down (a restart). */
export async function fetchServerVersion(): Promise<string | null> {
  try {
    const res = await fetch(`${BASE}/health`, { cache: "no-store" });
    if (!res.ok) return null;
    const body = (await res.json()) as { version?: unknown };
    return typeof body.version === "string" ? body.version : null;
  } catch {
    return null;
  }
}

// ── dataset explorer (ADR-165) ───────────────────────────────────────────────

const JSON_HEADERS = { "Content-Type": "application/json" };

export async function explorerSummary(
  form: ExplorerForm,
): Promise<ExplorerSummary> {
  return request<ExplorerSummary>("/explorer/summary", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(explorerBody(form)),
  });
}

export interface ExplorerImageItem {
  split: string;
  path: string;
  label_path: string | null;
  classes: string[];
  boxes: number;
}

export interface ExplorerImagesPage {
  total: number;
  offset: number;
  limit: number;
  items: ExplorerImageItem[];
}

export async function explorerImages(
  form: ExplorerForm,
  filter: { split: string | null; className: string | null },
  offset: number,
  limit: number,
): Promise<ExplorerImagesPage> {
  return request<ExplorerImagesPage>("/explorer/images", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({
      ...explorerBody(form),
      split: filter.split,
      class_name: filter.className,
      offset,
      limit,
    }),
  });
}

export async function explorerScanStart(
  form: ExplorerForm,
  maxImages: number,
): Promise<ScanJob> {
  return request<ScanJob>("/explorer/scan", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ ...explorerBody(form), max_images: maxImages }),
  });
}

export async function explorerScanStatus(jobId: string): Promise<ScanJob> {
  return request<ScanJob>(`/explorer/scan/${encodeURIComponent(jobId)}`);
}

export async function explorerScanCancel(jobId: string): Promise<void> {
  await request<unknown>(`/explorer/scan/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
  });
}

// ── incorporating images into a dataset (ADR-166) ────────────────────────────

const INCORPORATE = "/explorer/incorporate";

export async function incorporatePreview(
  form: ExplorerForm,
  sourceDir: string,
): Promise<IncorporateJob<IncorporatePreview>> {
  return request<IncorporateJob<IncorporatePreview>>(`${INCORPORATE}/preview`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ ...explorerBody(form), source_dir: sourceDir.trim() }),
  });
}

export async function incorporateStatus<R>(jobId: string): Promise<IncorporateJob<R>> {
  return request<IncorporateJob<R>>(`${INCORPORATE}/${encodeURIComponent(jobId)}`);
}

export async function incorporatePlan(
  jobId: string,
  body: PlanBody,
): Promise<IncorporatePlan> {
  return request<IncorporatePlan>(`${INCORPORATE}/${encodeURIComponent(jobId)}/plan`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });
}

export async function incorporateApply(
  jobId: string,
  body: PlanBody,
  destName: string,
): Promise<IncorporateJob<IncorporateResult>> {
  return request<IncorporateJob<IncorporateResult>>(
    `${INCORPORATE}/${encodeURIComponent(jobId)}/apply`,
    {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ ...body, dest_name: destName.trim() }),
    },
  );
}

export async function incorporateCancel(jobId: string): Promise<void> {
  await request<unknown>(`${INCORPORATE}/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
  });
}

// ── merging two datasets into a new one (ADR-167) ───────────────────────────

const MERGE = "/explorer/merge";

/** B is read with the same split folder names the form gives A. */
export async function mergePreview(
  form: ExplorerForm,
  otherDir: string,
): Promise<IncorporateJob<MergePreview>> {
  return request<IncorporateJob<MergePreview>>(`${MERGE}/preview`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({
      ...explorerBody(form),
      other_dir: otherDir.trim(),
      other_train_dir: form.train_dir,
      other_val_dir: form.val_dir,
      other_test_dir: form.test_dir,
    }),
  });
}

export async function mergeStatus<R>(jobId: string): Promise<IncorporateJob<R>> {
  return request<IncorporateJob<R>>(`${MERGE}/${encodeURIComponent(jobId)}`);
}

export async function mergePlan(jobId: string, mapping: MappingBody): Promise<MergePlan> {
  return request<MergePlan>(`${MERGE}/${encodeURIComponent(jobId)}/plan`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ mapping }),
  });
}

export async function mergeApply(
  jobId: string,
  mapping: MappingBody,
  destName: string,
): Promise<IncorporateJob<MergeResult>> {
  return request<IncorporateJob<MergeResult>>(`${MERGE}/${encodeURIComponent(jobId)}/apply`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ mapping, dest_name: destName.trim() }),
  });
}

export async function mergeCancel(jobId: string): Promise<void> {
  await request<unknown>(`${MERGE}/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
  });
}

// ── re-splitting a dataset into a new one (ADR-168) ─────────────────────────

const RESPLIT = "/explorer/resplit";

export async function resplitPreview(
  form: ExplorerForm,
  groupCsv: string,
): Promise<IncorporateJob<ResplitPreview>> {
  const csv = groupCsv.trim();
  return request<IncorporateJob<ResplitPreview>>(`${RESPLIT}/preview`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ ...explorerBody(form), ...(csv ? { group_csv: csv } : {}) }),
  });
}

export async function resplitStatus<R>(jobId: string): Promise<IncorporateJob<R>> {
  return request<IncorporateJob<R>>(`${RESPLIT}/${encodeURIComponent(jobId)}`);
}

export async function resplitPlan(jobId: string, body: ResplitBody): Promise<ResplitPlan> {
  return request<ResplitPlan>(`${RESPLIT}/${encodeURIComponent(jobId)}/plan`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  });
}

export async function resplitApply(
  jobId: string,
  body: ResplitBody,
  destName: string,
): Promise<IncorporateJob<ResplitResult>> {
  return request<IncorporateJob<ResplitResult>>(
    `${RESPLIT}/${encodeURIComponent(jobId)}/apply`,
    {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({ ...body, dest_name: destName.trim() }),
    },
  );
}

export async function resplitCancel(jobId: string): Promise<void> {
  await request<unknown>(`${RESPLIT}/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
  });
}

// ── converting between YOLO and COCO into a new dataset (ADR-169) ───────────

const COCO = "/explorer/coco";

type CocoSide = "export" | "import";

const cocoUrl = (side: CocoSide, rest: string) => `${COCO}/${side}${rest}`;
const cocoJob = (side: CocoSide, jobId: string, rest = "") =>
  cocoUrl(side, `/${encodeURIComponent(jobId)}${rest}`);

function cocoPost<T>(url: string, body?: unknown): Promise<T> {
  return request<T>(url, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body ?? {}),
  });
}

export async function cocoExportPreview(
  form: ExplorerForm,
): Promise<IncorporateJob<CocoExportPreview>> {
  return cocoPost(cocoUrl("export", "/preview"), { base_dir: form.base_dir });
}

export async function cocoExportStatus<R>(jobId: string): Promise<IncorporateJob<R>> {
  return request<IncorporateJob<R>>(cocoJob("export", jobId));
}

/** The preview again, with the destination name checked by the server. */
export async function cocoExportPlan(jobId: string, destName: string): Promise<unknown> {
  return cocoPost(cocoJob("export", jobId, "/plan"), { dest_name: destName.trim() });
}

export async function cocoExportApply(
  jobId: string,
  destName: string,
): Promise<IncorporateJob<CocoExportResult>> {
  return cocoPost(cocoJob("export", jobId, "/apply"), { dest_name: destName.trim() });
}

export async function cocoExportCancel(jobId: string): Promise<void> {
  await cocoPost<unknown>(cocoJob("export", jobId, "/cancel"));
}

export async function cocoImportPreview(
  sourceDir: string,
): Promise<IncorporateJob<CocoImportPreview>> {
  return cocoPost(cocoUrl("import", "/preview"), { source_dir: sourceDir.trim() });
}

export async function cocoImportStatus<R>(jobId: string): Promise<IncorporateJob<R>> {
  return request<IncorporateJob<R>>(cocoJob("import", jobId));
}

export async function cocoImportPlan(
  jobId: string,
  splits: Record<string, string>,
  destName?: string,
): Promise<CocoImportPlan> {
  return cocoPost(cocoJob("import", jobId, "/plan"), {
    splits,
    ...(destName ? { dest_name: destName } : {}),
  });
}

export async function cocoImportApply(
  jobId: string,
  splits: Record<string, string>,
  destName: string,
): Promise<IncorporateJob<CocoImportResult>> {
  return cocoPost(cocoJob("import", jobId, "/apply"), {
    splits,
    dest_name: destName.trim(),
  });
}

export async function cocoImportCancel(jobId: string): Promise<void> {
  await cocoPost<unknown>(cocoJob("import", jobId, "/cancel"));
}


export interface HubTokenStatus {
  configured: boolean;
  shared: boolean;
}

export interface HubPreview {
  files: string[];
  digest: string;
  card: string;
}

/** Read this profile's token availability without returning the secret. */
export async function fetchHubToken(): Promise<HubTokenStatus> {
  return request<HubTokenStatus>("/hub/token");
}

/** Save a Hub token for the active profile on the server. */
export async function saveHubToken(token: string): Promise<HubTokenStatus> {
  return request<HubTokenStatus>("/hub/token", {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
}

/** Remove the active profile's saved Hub token. */
export async function deleteHubToken(): Promise<HubTokenStatus> {
  return request<HubTokenStatus>("/hub/token", { method: "DELETE" });
}

/** Preview the exact files and card before an explicit Hub upload. */
export async function fetchHubPreview(runId: string): Promise<HubPreview> {
  return request<HubPreview>(`/hub/preview?run_id=${encodeURIComponent(runId)}`);
}

/** Upload the previously reviewed snapshot to the requested Hub repository. */
export async function exportRunToHub(runId: string, repoId: string, digest: string): Promise<{ url: string }> {
  return request<{ url: string }>("/hub/export", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId, repo_id: repoId, digest }),
  });
}

/** Which Zenodo receives a draft: the real one or its test sandbox (ADR-174). */
export type ZenodoSite = "zenodo" | "sandbox";

export interface ZenodoPreview {
  name: string;
  size: number;
  files: string[];
  notes: { artifact: string; reason: string }[];
  digest: string;
}

export interface ZenodoDraft {
  id: number | null;
  url: string;
  sandbox: boolean;
  name: string;
}

const zenodoToken = (site: ZenodoSite) => (site === "sandbox" ? "/zenodo/sandbox/token" : "/zenodo/token");

/** Whether this profile has a token for `site`; the token never comes back. */
export async function fetchZenodoToken(site: ZenodoSite): Promise<HubTokenStatus> {
  return request<HubTokenStatus>(zenodoToken(site));
}

/** Save this profile's token for `site` on the server. */
export async function saveZenodoToken(site: ZenodoSite, token: string): Promise<HubTokenStatus> {
  return request<HubTokenStatus>(zenodoToken(site), {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
}

/** Forget this profile's token for `site`. */
export async function deleteZenodoToken(site: ZenodoSite): Promise<HubTokenStatus> {
  return request<HubTokenStatus>(zenodoToken(site), { method: "DELETE" });
}

/** The bundle a draft would receive, built on the server without the network. */
export async function fetchZenodoPreview(runId: string): Promise<ZenodoPreview> {
  return request<ZenodoPreview>(`/zenodo/preview?run_id=${encodeURIComponent(runId)}`);
}

/** Create an unpublished draft holding the previewed bundle. */
export async function createZenodoDraft(runId: string, digest: string, site: ZenodoSite): Promise<ZenodoDraft> {
  return request<ZenodoDraft>("/zenodo/export", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId, digest, sandbox: site === "sandbox" }),
  });
}

export interface GitHubPreview {
  folder: string;
  files: { path: string; size: number }[];
  readme: string;
  digest: string;
}

export interface GitHubCommit {
  url: string;
  commit: string;
  branch: string;
  folder: string;
}

/** Whether this profile has a GitHub token; the token never comes back. */
export async function fetchGitHubToken(): Promise<HubTokenStatus> {
  return request<HubTokenStatus>("/github/token");
}

/** Save this profile's GitHub token on the server. */
export async function saveGitHubToken(token: string): Promise<HubTokenStatus> {
  return request<HubTokenStatus>("/github/token", {
    method: "PUT", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
}

/** Forget this profile's GitHub token. */
export async function deleteGitHubToken(): Promise<HubTokenStatus> {
  return request<HubTokenStatus>("/github/token", { method: "DELETE" });
}

/** The folder, files and README a commit would hold, built offline (ADR-175). */
export async function fetchGitHubPreview(runId: string): Promise<GitHubPreview> {
  return request<GitHubPreview>(`/github/preview?run_id=${encodeURIComponent(runId)}`);
}

/** Commit the previewed files to a private repository. */
export async function commitRunToGitHub(runId: string, repo: string, digest: string): Promise<GitHubCommit> {
  return request<GitHubCommit>("/github/export", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId, repo, digest }),
  });
}

export type DvcRepositoryStateName = "ready" | "missing" | "not_dir" | "not_git" | "no_dvc";

export interface DvcRepositoryState {
  path: string;
  state: DvcRepositoryStateName;
  remote: string | null;
  tools: { git: boolean; dvc: boolean };
}

export interface DvcPreview {
  folder: string;
  files: { path: string; size: number; storage: "git" | "dvc" }[];
  readme: string;
  digest: string;
  repository: DvcRepositoryState;
}

export interface DvcExportResult {
  folder: string;
  path: string;
  commit: string | null;
  remote: string | null;
  pushed: boolean | null;
}

/** The folder, its files (git or DVC) and the repository's state (ADR-176). */
export async function fetchDvcPreview(runId: string, repo: string): Promise<DvcPreview> {
  const query = `run_id=${encodeURIComponent(runId)}&repo=${encodeURIComponent(repo)}`;
  return request<DvcPreview>(`/dvc/preview?${query}`);
}

/** Make the folder a git repository with DVC (git init, dvc init). */
export async function prepareDvcRepository(repo: string): Promise<DvcRepositoryState> {
  return request<DvcRepositoryState>("/dvc/prepare", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo }),
  });
}

/** Write the previewed folder, dvc add the weights, commit it and push them. */
export async function versionRunWithDvc(runId: string, repo: string, digest: string): Promise<DvcExportResult> {
  return request<DvcExportResult>("/dvc/export", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_id: runId, repo, digest }),
  });
}
