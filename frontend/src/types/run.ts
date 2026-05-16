export interface RunStatus {
  status: "idle" | "running" | "completed" | "failed";
  run_id: string | null;
  error: string | null;
}

export interface RunResponse {
  run_id: string;
  status: "running";
}

export interface RunResult {
  run_id: string;
  metrics: Record<string, number | null>;
  report: Record<string, unknown>;
  artifacts: {
    model?: string;
    graphics?: string[];
    report?: string | null;
  };
}
