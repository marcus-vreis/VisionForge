import type { JsonSchema } from "../types/schema";
import type { RunResponse, RunResult, RunStatus, RunSummary } from "../types/run";

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

export async function fetchStatus(): Promise<RunStatus> {
  return request<RunStatus>("/experiment/status");
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

export interface RunDetail {
  run_id: string;
  experiment_name: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  device_used: string | null;
  run_dir: string;
  config: Record<string, unknown>;
  metrics: Record<string, unknown>;
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

export function artifactUrl(path: string): string {
  return `${BASE}/artifacts/${path}`;
}
