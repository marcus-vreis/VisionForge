import type { JsonSchema } from "../types/schema";
import type { RunResponse, RunResult, RunStatus, RunSummary } from "../types/run";

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.detail ?? res.statusText);
  }
  return res.json();
}

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
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

export function artifactUrl(path: string): string {
  return `${BASE}/artifacts/${path}`;
}

export async function fetchRuns(): Promise<RunSummary[]> {
  return request<RunSummary[]>("/runs");
}
