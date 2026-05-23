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

/** Summary of one historical experiment run, matching backend RunSummary. */
export interface RunSummary {
  run_id: string;
  experiment_name: string;
  model_arch: string;
  task: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  epochs_completed: number;
  final_metrics: Record<string, number>;
  /** Number of preprocessing filters applied during training (0 = none). */
  preprocessing_count?: number;
}

/** Discriminated union of SSE events emitted by GET /api/experiment/events. */
export type TrainingEvent =
  | { event: "start"; total_epochs: number }
  | {
      event: "epoch_end";
      epoch: number;
      total_epochs: number;
      train_loss: number;
      val_loss: number;
      val_accuracy: number;
      elapsed_s: number;
    }
  | { event: "end"; total_epochs: number };
