import type { Dict } from "./pt";

/** English. Typed as the Portuguese dictionary, so the two cannot drift. */
export const en: Dict = {
  common: {
    back: "Back",
    next: "Continue",
    skip: "Skip",
    close: "Close",
    cancel: "Cancel",
    save: "Save",
    loading: "Loading…",
  },
  header: {
    guide: "guide",
    guideTitle: "Show the interface guide again",
    changeName: "Change name",
    welcome: "Welcome,",
  },
  language: {
    label: "Language",
    switchTo: "Switch to Portuguese",
  },
  errors: {
    cannotConnect: "Could not reach the server. Check that the backend is running.",
    validation: "There are validation errors in the form.",
    markdownExport: (status: number) => `Markdown export failed (HTTP ${status}).`,
    deviceInfo: "Failed to load devices.",
  },
  app: {
    train: {
      simple: "▶ Train",
      cv: "▶ Run K-fold",
      sweep: "▶ Run sweep",
      replicates: "▶ Run replicates",
    },
    error: "Error",
    unnamedRun: "training",
  },
  bottomBar: {
    reopenTraining: "🔬 open training",
    running: "Running…",
    history: "history",
    datasets: "datasets",
    datasetsTitle: "Download a dataset to a local folder",
    queue: "queue",
    queueTitle: "View and reorder the runs waiting for the GPU",
  },
  deviceSelector: {
    using: "using",
    loadingTitle: "Loading devices…",
    cudaUnavailableTitle: "CUDA unavailable — CPU only",
    cudaTitle: (version: string, gpus: number) =>
      `CUDA ${version} · ${gpus} ${gpus === 1 ? "GPU" : "GPUs"}`,
    cudaSummary: (version: string, gpus: number) =>
      `cuda ${version} · ${gpus} ${gpus === 1 ? "gpu" : "gpus"}`,
    gpuName: (index: number, name: string) => `GPU ${index} · ${name}`,
    multiGpu: (gpus: number) => `Multi-GPU (${gpus} GPUs)`,
    loadingShort: "loading…",
    errorPrefix: (message: string) => `error: ${message}`,
    cudaNotDetected: "cuda not detected",
    cpuSubtitle: "Train on the CPU",
  },
  welcome: {
    hello: "Welcome",
    helloName: (name: string) => `Welcome, ${name}`,
    askName: "What's your name?",
    placeholder: "type here",
    nameLabel: "Your name",
    enter: "Enter ↵",
  },
  datasetsOverlay: {
    kicker: "// datasets",
    title: "Get data",
    description:
      "Download once to a local folder, then point any task's dataset field at it.",
  },
  lightbox: {
    closeTitle: "Close (Esc)",
  },
  taskHero: {
    kicker: (short: string) => `// task / ${short}`,
    dataset: "Dataset",
    task: "Task",
  },
  experimentRunner: {
    inProgress: "Training in progress...",
    runId: "Run ID:",
  },
  advancedFields: {
    advanced: "advanced",
    changed: "values changed",
  },
  modelAdvice: {
    measured: (architecture: string, optimizer: string, learningRate: number) =>
      `For ${architecture}, the measured setting is ${optimizer} at ${learningRate}.`,
    apply: (optimizer: string, learningRate: number) =>
      `use ${optimizer} · ${learningRate}`,
  },
  infoDot: {
    label: "Explanation",
  },
  workersField: {
    label: "Workers",
    setManually: "Set the number manually",
    backToAuto: "Go back to automatic, based on the machine's free memory",
    automatic: "automatic",
    suggestedNow: (n: number) => `≈ ${n} now`,
  },
  paramHelp: {
    // ── basic ─────────────────────────────────────────────────────────────────
    epochs:
      "How many times the model sees the whole dataset. More epochs learn more, until it starts memorizing.",
    batch_size:
      "How many images per step. Larger gives a steadier gradient and uses more VRAM; if you run out of memory, lower this one first.",
    learning_rate:
      "The size of each update step. Too high and it diverges, too low and it never gets there.",
    seed:
      "Fixes the randomness (initial weights, data order). The same seed on the same data gives the same result.",

    // ── advanced: optimization ────────────────────────────────────────────────
    optimizer:
      "The algorithm that applies the gradient. adam converges fast without fine tuning; sgd usually generalizes better if you give it time.",
    momentum:
      "How much the previous step carries into the current one. Smooths the trajectory and helps cross plateaus.",
    weight_decay:
      "Pulls the weights toward zero. Fights overfitting; too high and the model can't learn.",
    learning_rate_final: "Fraction of the initial learning rate at the end of training.",
    lrf: "Fraction of the initial learning rate at the end of training.",

    // ── advanced: scheduling ──────────────────────────────────────────────────
    scheduler:
      "How the learning rate decreases during training. Letting it decay almost always helps.",
    step_size: "How many epochs between learning rate reductions.",
    gamma: "What the learning rate is multiplied by at each reduction.",
    cos_lr: "Makes the learning rate follow a cosine curve instead of dropping in steps.",
    warmup_epochs:
      "Initial epochs with the learning rate ramping up slowly, so the model doesn't destabilize at the start.",

    // ── advanced: stopping and regularization ─────────────────────────────────
    early_stopping_patience:
      "Consecutive epochs without improvement before training stops. Leave at 0 (or empty) to run all configured epochs.",
    patience: "Epochs without improvement before it stops on its own.",
    label_smoothing: "Softens the labels so the model doesn't become overconfident.",
    dropout:
      "Randomly switches off neurons during training, so the model can't rely on just a few.",
    freeze: "Freezes the first N layers. Useful for transfer learning with little data.",

    // ── advanced: mechanics ───────────────────────────────────────────────────
    amp:
      "Mixed precision: uses 16 bits where it can. Trains faster and uses less VRAM, with a low risk of instability.",
    mixed_precision:
      "Does part of the math in 16 bits. Speeds up training and uses less VRAM on recent GPUs; on sensitive models it can cost numerical precision.",
    deterministic:
      "Makes the same config with the same seed return exactly the same numbers. On by default: we measured the cost, and it is zero or negative on short runs.",
    base_dir:
      "Root folder of the dataset. Inside it are the train, validation and test subfolders — VisionForge looks for the usual names (train/val/test, treino/validacao/teste) and fills them in on its own when it finds them. What each subfolder holds depends on the task: one folder per class for classification, images and labels for detection, images and masks for segmentation.",
    train_dir:
      "Subfolder used to fit the weights. It's the only one the model sees during training.",
    val_dir:
      "Subfolder used every epoch to measure progress and pick the best checkpoint. It never goes into fitting the weights.",
    test_dir:
      "Subfolder evaluated a single time, at the end. It lets you report a result that hasn't influenced any decision.",
    coreset_ratio:
      "How much of the \"normal\" PatchCore keeps to compare against later. It cuts the training images into small patches and keeps a sample of them, as varied as possible; a new image is anomalous when some patch of it looks like nothing that was kept. 1% is the original paper's value. Raising it makes the memory bank more complete, but build time grows in the same proportion: 10% takes ten times longer.",
    num_workers:
      "Processes that load the images in parallel. On automatic, VisionForge divides the machine's free memory by the cost of one worker — on Windows each one reloads torch and the CUDA DLLs, ~1 GB, and a number that is too high doesn't make training slow: it keeps training from starting (WinError 1455).",
    workers:
      "Processes that load the images in parallel. On automatic, VisionForge divides the machine's free memory by the cost of one worker — on Windows each one reloads torch and the CUDA DLLs, ~1 GB, and a number that is too high doesn't make training slow: it keeps training from starting (WinError 1455).",
    pin_memory: "Speeds up copying the images to the GPU. Leave it on unless you're short on RAM.",
    image_size:
      "Training resolution. Larger sees more detail, and the VRAM and time it costs grow with the square.",
    nbs: "Nominal batch size used to normalize weight decay when the real batch is smaller.",
    single_cls:
      "Treats all classes as one. Use it to measure only how well the boxes are localized.",
    rect:
      "Groups images with similar aspect ratios instead of forcing squares. Faster, less uniform.",
    multi_scale:
      "Varies the resolution between steps, so the model copes with objects of different sizes.",
    close_mosaic:
      "Turns mosaic off for the last N epochs, so the model finishes training on real images.",
    box: "Weight of the box localization loss.",
    cls: "Weight of the classification loss.",
    dfl: "Weight of the loss on the distribution of the box edges.",
  } as Record<string, string>,
  paramPanel: {
    sectionLabels: {
      model: "Model",
      training: "Training",
      data: "Dataset",
      output: "Output",
      classification: "Classification",
      transforms: "Transforms",
    },
    fieldLabels: {
      "model.name": "Architecture",
      name: "Experiment name",
      task: "Task type",
      block: "Block",
      num_classes: "Number of classes",
      pretrained: "Pretrained weights",
      weights_path: "Weights path",
      learning_rate: "Learning Rate",
      epochs: "Epochs",
      batch_size: "Batch size",
      early_stopping_patience: "Early stop (patience)",
      optimizer: "Optimizer",
      weight_decay: "Weight decay",
      seed: "Seed",
      deterministic: "Deterministic",
      mixed_precision: "Mixed precision (AMP)",
      kind: "Type",
      step_size: "Step size",
      gamma: "Gamma",
      patience: "Patience",
      factor: "Factor",
      min_lr: "Min LR",
      base_dir: "Base directory",
      train_dir: "Train subdir",
      val_dir: "Validation subdir",
      test_dir: "Test subdir",
      num_workers: "Workers",
      pin_memory: "Pin memory",
      image_size: "Image size",
      horizontal_flip: "Horizontal flip",
      rotation_degrees: "Rotation (degrees)",
      color_jitter: "Color jitter",
      normalize_mean: "Normalization (mean)",
      normalize_std: "Normalization (std)",
      n_folds: "Number of folds",
      stratified: "Stratified",
      shuffle: "Shuffle",
      fold_seed: "Fold seed",
      mode: "Mode",
      unfreeze_from_layer: "Unfreeze from layer",
      backbone_lr_multiplier: "Backbone LR (×)",
      model_names: "Architectures",
      metric: "Ranking metric",
    },
    kickers: {
      strategy: "// experiment strategy",
      scheduler: "// learning-rate scheduler",
      crossValidation: "// k-fold cross-validation",
      transferLearning: "// transfer learning",
      model: "// model",
      training: "// training",
      dataset: "// dataset",
      classes: "// classes",
      image: "// image",
      augmentation: "// data augmentation",
      comingSoon: "// coming soon",
    },
    blocks: {
      simple: "Simple training",
      crossValidation: "K-Fold (CV)",
      transferLearning: "Transfer learning",
      gridSearch: "Grid search",
      randomSearch: "Random search",
    },
    blockHints: {
      crossValidation:
        "Trains N models on N folds of the training folder. Validation per fold; normalize_mean/std are recomputed per fold to avoid data leakage. Doesn't use the test split — results are aggregated in `cv_summary.json`.",
      transferLearning:
        "Feature extraction (trains only the head) or fine-tuning (head + part of the backbone with a smaller LR). Useful on small datasets, without destroying the pretrained features. In feature extraction the backbone weights don't move, but the BatchNorm statistics recalibrate to your dataset — the backbone is frozen, not identical.",
      gridSearch:
        'Trains **once per combination** of the Cartesian product of the space defined below. Each key is a dot-path (e.g. `training.learning_rate`); the value is a list. Careful: 3×3×2 is already 18 runs. To **compare architectures**, add values to the "Architecture" field ("+ add to grid" button) — a single-axis grid; compare the runs in the history.',
      randomSearch:
        "Samples `n_trials` independent configurations from the space below. Each parameter has a type: `uniform`, `log_uniform` (LR and weight_decay) or `choice` (discrete lists).",
    },
    weights: {
      label: "Custom checkpoint (.pth)",
      clearTitle: "Remove custom checkpoint (go back to pretrained / random weights)",
      clear: "clear",
      placeholder: "optional — overrides ImageNet",
      browse: "Browse",
      cancelled: "Cancelled.",
      pickFailed: "Failed to open the file picker.",
    },
    grid: {
      addValue: "+ add to grid",
      addAnother: "+ value",
      removeValue: "Remove value",
      axisTag: (values: number) => `grid · ${values} values`,
      banner:
        "**Grid search is on.** Click `+ add to grid` on any __Model__ or __Training__ hyperparameter to sweep several values.",
      trials: (n: number) =>
        `// ${n} trial${n === 1 ? "" : "s"}${n > 12 ? " ⚠️ high" : ""}`,
    },
    randomSearch: {
      kicker: (trials: number) =>
        `// random search · ${trials} trial${trials === 1 ? "" : "s"}`,
      add: "+ add",
      trialsLabel: "n_trials",
      seedLabel: "seed",
      empty: 'Search space is empty — click "+ add".',
      example: "Example: `training.learning_rate` = log_uniform(1e-5, 1e-2).",
      keyPlaceholder: "dot-path (e.g. training.learning_rate)",
      choicesPlaceholder: "csv: resnet18, resnet50",
      low: "low",
      high: "high",
      removeRow: "Remove row",
    },
    modelComparison: {
      kicker: (selected: number) => `// model comparison · ${selected} selected`,
      needTwo: "Select at least 2 architectures to start the comparison.",
    },
    lockedClasses: {
      title: "Binary task — fixed at 1",
      badge: "🔒 binary",
    },
    normalizePlaceholder: "e.g. 0.485, 0.456, 0.406",
    exportYaml: "↓ Export YAML",
    exportTitle: "Export the current configuration as a .yaml file",
    importYaml: "↑ Import YAML",
    importTitle: "Import a configuration from a .yaml file",
    importWarnings: (count: number, summary: string, extra: number) =>
      `YAML imported with ${count} structural ${count === 1 ? "warning" : "warnings"}:\n${summary}${extra > 0 ? `\n…(+${extra} more)` : ""}\n\nFix these before training — the backend rejects the config on Pydantic validation.`,
    unavailable: (task: string) => `${task} is not available yet`,
    unavailableBody: "This task will be implemented in an upcoming phase of VisionForge.",
    loadingSchema: "loading schema…",
    hiddenParams: (n: number) => `${n} hidden parameters — turn it on to adjust`,
    fieldErrors: (n: number) => `${n} ${n === 1 ? "field" : "fields"} with errors:`,
  },
};
