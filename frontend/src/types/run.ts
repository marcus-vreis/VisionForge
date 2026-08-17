export interface RunStatus {
  /** `queued` is client-side only: the server reports the *active* run, and the
   *  hook derives this for a submission of its own that has not started yet. */
  status: "idle" | "queued" | "running" | "completed" | "failed";
  run_id: string | null;
  error: string | null;
  /** Submissions waiting behind the active run (ADR-075). */
  queued?: number;
  /** 1-based place in the queue while this run is waiting. */
  position?: number;
}

export interface RunResponse {
  run_id: string;
  /** `queued` when the GPU was busy and the job is waiting its turn (ADR-075). */
  status: "running" | "queued";
}

/** One entry in the run queue. */
export interface QueuedJobInfo {
  run_id: string;
  label: string;
  task: string;
  strategy: string;
  submitted_at: string;
}

export interface QueueSnapshot {
  active: QueuedJobInfo | null;
  pending: QueuedJobInfo[];
}

/** Bootstrap interval for one test metric (backend MetricCI, ADR-074). */
export interface MetricCI {
  metric: string;
  value: number;
  ci_low: number;
  ci_high: number;
  confidence: number;
  n_resamples: number;
  n_samples: number;
}

export interface RunResult {
  run_id: string;
  metrics: Record<string, number | null>;
  /** Keyed by bare metric name (`accuracy`), while `metrics` prefixes the
   *  test-split entries (`test_accuracy`). Absent on runs trained before
   *  ADR-074 and on tasks that keep no per-sample predictions. */
  metric_cis?: Record<string, MetricCI>;
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
  /** Block type from config — distinguishes classification / cross_validation /
   *  grid_search / etc. in the history list. */
  block?: string;
  /** Dataset the run trained on. Derived server-side, falling back to
   *  config.data.base_dir so it resolves on runs older than the fingerprint. */
  dataset_name?: string | null;
  dataset_root?: string | null;
  /** True when the run stopped before its last epoch and left state behind
   *  (ADR-092/093). Derived server-side from what is on disk. */
  resumable?: boolean;
  /** How many epochs it was configured to run, so the button can say how many
   *  are missing. */
  configured_epochs?: number | null;
}

/** The dataset a run trained on, as far as its run.json can prove it.
 *
 * `name` and `root` exist for every run; everything below comes from the
 * fingerprint (ADR-061) and is absent on runs written before 2026-07-26.
 */
export interface DatasetInfo {
  name: string;
  root: string;
  n_files?: number | null;
  total_bytes?: number | null;
  method?: string | null;
  digest?: string | null;
  note?: string | null;
}

/** Discriminated union of SSE events emitted by GET /api/experiment/events.
 *
 * Multi-trial blocks (grid_search, random_search) wrap each inner training in
 * a trial_start/trial_end pair and annotate epoch_end with trial_index /
 * total_trials so the overlay can show "trial k/N" progress. A single terminal
 * "end" closes the stream once the whole sweep finishes. */
export type TrainingEvent =
  | { event: "start"; total_epochs: number; device?: string }
  | {
      event: "trial_start";
      trial_index: number;
      total_trials: number;
      overrides: Record<string, unknown>;
      seed: number;
    }
  | {
      event: "epoch_end";
      epoch: number;
      total_epochs: number;
      train_loss: number;
      // Optional: the generic engine behind researcher-defined tasks
      // (ADR-058) has no notion of a validation loss and streams the task's
      // own declared metrics as `val_<name>` instead. val_accuracy is the
      // compat field every task fills so the live chart has a series.
      val_loss?: number;
      val_accuracy: number;
      elapsed_s?: number;
      // Custom tasks' declared metrics arrive as val_<name>.
      [customMetric: `val_${string}`]: number | null | undefined | string;
      trial_index?: number;
      total_trials?: number;
      // Detection-specific metrics (present only on detection runs). All
      // optional so the shared overlay degrades cleanly for other tasks.
      map50?: number | null;
      map50_95?: number | null;
      precision?: number | null;
      recall?: number | null;
      box_loss?: number | null;
      train_box_loss?: number | null;
      train_cls_loss?: number | null;
      train_dfl_loss?: number | null;
      val_box_loss?: number | null;
      val_cls_loss?: number | null;
      val_dfl_loss?: number | null;
    }
  | {
      event: "trial_end";
      total_epochs: number;
      trial_index: number;
      total_trials: number;
    }
  | { event: "end"; total_epochs: number; total_trials?: number };
