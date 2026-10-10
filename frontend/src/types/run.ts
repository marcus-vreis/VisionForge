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

/** Where a job stops when asked (ADR-111): the boundary after which it starts
 *  nothing new. `phase` is PatchCore, which has no epochs. */
export type StopPoint =
  | "epoch"
  | "trial"
  | "fold"
  | "model"
  | "replicate"
  | "phase";

/** One entry in the run queue. */
export interface QueuedJobInfo {
  run_id: string;
  label: string;
  task: string;
  strategy: string;
  submitted_at: string;
  /** Where the job stops when asked; `null` is a job that cannot be stopped once
   *  running (DELETE answers 409). Absent from servers older than ADR-111. */
  stop_at?: StopPoint | null;
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
  /** True when a user stop cut this run (ADR-111): a single run whose epoch loop
   *  broke on the stop, or a multi-unit job with a unit cut or units left unrun
   *  — including one stopped before any unit finished, which ends `completed`
   *  with no mean. Absent from servers older than that; an error during the stop
   *  window is a failed run, never a stopped one. */
  stopped?: boolean;
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
  /** The dataset fingerprint (ADR-061) behind that name, when the run has one:
   *  the ranking groups by it, since the same path may hold different files. The
   *  digest only means something next to its `dataset_method`. Absent or null on
   *  a run older than the fingerprint. */
  dataset_digest?: string | null;
  dataset_method?: string | null;
  /** A user stop cut this run (ADR-111): listed in the ranking, never ranked. */
  stopped?: boolean;
  /** Which way each of `final_metrics` improves, by the server's rule. Absent
   *  from a server that predates it: the page then reads the metric's name. */
  metric_directions?: Record<string, "higher" | "lower">;
  /** True when the run stopped before its last epoch and left state behind
   *  (ADR-092/093). Derived server-side from what is on disk. */
  resumable?: boolean;
  /** How many epochs it was configured to run, so the button can say how many
   *  are missing. */
  configured_epochs?: number | null;
  /** Set on a run that is one seed of a replicate group: the group's run id
   *  (ADR-113). History folds it under the group. */
  group_id?: string | null;
  /** Set on the group itself, which History lists as one run (ADR-113). */
  group?: RunGroupBrief | null;
}

/** How a replicate job is told apart: a set of seeds of one config, or one
 *  set of seeds per variant with paired tests between them. */
export type GroupKind = "replicates" | "replicated_comparison";

/** Distribution of one metric over a group's seeds, as the report recorded it
 *  (`aggregate_replicates`). Copied by the server, never recomputed here. */
export interface MetricAggregate {
  /** Seeds that ran to the end and reported the metric. */
  n: number;
  mean: number | null;
  /** Sample standard deviation (divides by n−1, `std_ddof`); null below two seeds. */
  std: number | null;
  std_ddof?: number;
  min: number | null;
  max: number | null;
  /** Student-t 95% interval of the mean; null below two seeds. */
  ci95_low: number | null;
  ci95_high: number | null;
  boot95_low?: number | null;
  boot95_high?: number | null;
}

/** The list view of a group (backend `RunGroupBrief`). */
export interface RunGroupBrief {
  kind: GroupKind;
  metric: string | null;
  /** The seeds asked for. */
  seeds: number[];
  /** Trainings asked for and trainings that ran to the end. */
  n_requested: number;
  n_finished: number;
  stopped: boolean;
  /** Run ids of the seeds' own runs that exist on disk. */
  child_ids: string[];
  /** Replicates: the aggregate behind each number of `final_metrics`, same key. */
  final_aggregates: Record<string, MetricAggregate>;
  /** Replicated comparison: the variants and the one with the best mean. */
  variants: string[];
  best_by_mean: string | null;
}

/** One seed of a group. `run_id` is null for a seed that never got a run. */
export interface GroupChild {
  seed: number;
  variant: string | null;
  /** `success`, `failed` or `stopped` (see lib/unit-status). */
  status: string;
  run_id: string | null;
  run_dir: string | null;
  metrics: Record<string, number | null>;
  error: string;
}

/** One paired test between two variants (backend `PairedComparison`). */
export interface PairedTest {
  label_a: string;
  label_b: string;
  metric: string;
  n_pairs: number;
  mean_a: number | null;
  mean_b: number | null;
  /** a − b */
  mean_difference: number | null;
  test: "paired_t" | "wilcoxon";
  test_reason: string;
  /** Raw p, before the Holm correction `significant` already carries. */
  p_value: number | null;
  /** Paired Cohen's d. */
  effect_size: number | null;
  effect_label: string;
  /** Smallest p this test could return with this many pairs. */
  min_achievable_p: number;
  /** True when that floor is above alpha: no result could be significant. */
  underpowered: boolean;
  /** After the Holm correction over the whole family of tests. */
  significant: boolean;
}

export interface GroupVariant {
  overrides: Record<string, unknown>;
  aggregates: Record<string, MetricAggregate>;
  successful: number | null;
  seeds_finished: number[];
  children: GroupChild[];
}

/** The `group` section of a group's run.json, as the run detail serves it. */
export interface RunGroup {
  kind: GroupKind;
  metric: string | null;
  seeds: number[];
  n_requested: number;
  n_finished: number;
  stopped: boolean;
  /** The divisor every `std` in the group used: 1 is the sample std (n-1). */
  std_ddof: number;
  report_path: string | null;
  report_dir: string | null;
  // Replicate set.
  seeds_finished?: number[];
  children?: GroupChild[];
  aggregates?: Record<string, MetricAggregate>;
  /** run.json `metrics` key -> the aggregate that filled it. */
  metric_keys?: Record<string, string>;
  // Replicated comparison.
  metric_direction?: "higher" | "lower" | null;
  alpha?: number | null;
  variants?: Record<string, GroupVariant>;
  comparisons?: PairedTest[];
  best_by_mean?: string | null;
  ranked_by_mean?: string[];
  /** The seeds every ranked variant finished; empty when nothing can be ranked. */
  ranking_seeds?: number[];
  significant_pairs?: number | null;
  skipped_variants?: string[];
  /** Variants a stop kept from starting. */
  not_run?: string[];
  underpowered?: boolean;
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
  // Trabalho longo que não é uma época. O PatchCore não tem épocas e leva de
  // minutos a horas montando o banco de memória; sem isso a tela ficava parada
  // o tempo todo e o treino parecia travado.
  | {
      event: "phase";
      label: string;
      done: number;
      total: number;
      elapsed_s?: number;
    }
  | { event: "end"; total_epochs: number; total_trials?: number };
