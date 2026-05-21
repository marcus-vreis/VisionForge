import type { JsonSchema } from "../types/schema";
import type { RunResponse, RunResult, RunStatus } from "../types/run";

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

export function artifactUrl(path: string): string {
  return `${BASE}/artifacts/${path}`;
}
