import type { CollapseFacts, Dict, LrPlanEvidence } from "./pt";

// Help texts shared by two field names; see the same pair in pt.ts.
const workersHelp =
  "DataLoader processes that load images in parallel. When set to automatic, VisionForge divides the machine's free memory by the cost of one worker. On Windows each worker reloads torch and the CUDA DLLs (~1 GB), so too high a value doesn't make training slow: it keeps training from starting (WinError 1455).";
const lrFinalHelp =
  "The learning rate at the end of training, as a fraction of the initial one.";

// What the collapse note says about the suggested setting (components/ModelAdvice). "Trains
// normally" is a claim, so it is only ever a number, and only for the model whose recovery was run
// on the same setup; everywhere else the setting is just suggested.
const suggestedRemedy = (
  recoveredAccuracy: number | null,
  optimizer: string,
  learningRate: number,
): string =>
  recoveredAccuracy === null
    ? `We suggest ${optimizer} at ${learningRate}.`
    : `With ${optimizer} at ${learningRate}, accuracy was ${recoveredAccuracy.toFixed(2)} under the same conditions.`;

// The measured part of the collapse note. Every number was measured on classification, so on any
// other form the note says "In classification" and nothing more about the task (a VGG regressor was
// run too, ADR-101, so "this task was not measured" would be false), and it never carries the
// recovery over: "under the same conditions" would not be true there.
const measuredNote = (facts: CollapseFacts, whatHappened: string): string => {
  const { architecture, measuredOn, accuracy, recoveredAccuracy, optimizer, learningRate } = facts;
  const result = `(accuracy ${accuracy.toFixed(2)} on 4 classes)`;
  const measuredItself = architecture.toLowerCase() === measuredOn;
  if (facts.task !== "classification") {
    const finding = measuredItself
      ? `In classification, ${measuredOn} ${whatHappened} with Adam at 1e-3 ${result}.`
      : `In classification, ${measuredOn}, from the same family as ${architecture}, ${whatHappened} with Adam at 1e-3 ${result}; ${architecture} was not measured.`;
    return `${finding} ${suggestedRemedy(null, optimizer, learningRate)}`;
  }
  const finding = measuredItself
    ? `${architecture} ${whatHappened} with Adam at 1e-3 ${result}.`
    : `${architecture}: ${measuredOn}, from the same family, ${whatHappened} with Adam at 1e-3 ${result}; this model was not measured.`;
  return `${finding} ${suggestedRemedy(measuredItself ? recoveredAccuracy : null, optimizer, learningRate)}`;
};

/** 1st, 2nd, 3rd, 4th…: a place in the run queue. */
const ordinal = (n: number): string => {
  const lastTwo = n % 100;
  if (lastTwo >= 11 && lastTwo <= 13) return `${n}th`;
  const suffixes: Record<number, string> = { 1: "st", 2: "nd", 3: "rd" };
  return `${n}${suffixes[n % 10] ?? "th"}`;
};

/** English. Typed as the Portuguese dictionary, so the two cannot drift. */
export const en: Dict = {
  evidence: {
    audit: "Audit images",
    auditing: "Auditing images…",
    title: "Split audit",
    scope: "Exact duplicates by SHA-256. Visual similarity is not checked.",
    clean: "No findings in the audited splits.",
    findings: "The audit found problems.",
    failed: "Failed to audit the images.",
    groupManifest: "Group CSV (optional)",
    groupHint: "Columns path,group; paths relative to the dataset root. In K-fold, whole groups stay in the same fold.",
    groupPlaceholder: "groups.csv",
    strictGroups: "Require a group for every image",
    strictHint: "Use with a group CSV to detect images without a group.",
    duplicateCount: (n: number) => `Exact duplicate groups: ${n}`,
    missingCount: (n: number) => `Missing splits: ${n}`,
    emptyCount: (n: number) => `Empty splits: ${n}`,
    unreadableCount: (n: number) => `Unreadable images: ${n}`,
    groupCount: (n: number) => `Groups crossing splits: ${n}`,
    imageCount: (split: string, n: number) => `${split}: ${n} images`,
    details: "Details (up to 5 findings per category, 5 paths per group)",
    bundle: "Download experiment bundle",
    bundling: "Preparing bundle…",
    bundleFailed: "Failed to download the bundle.",
  },
  common: {
    back: "Back",
    skip: "Skip",
    close: "Close",
    cancel: "Cancel",
  },
  // The five task families by name: the history tabs and the queue rows.
  taskNames: {
    classification: "Classification",
    detection: "Detection",
    regression: "Regression",
    segmentation: "Segmentation",
    anomaly: "Anomaly",
  },
  // The button that downloads a run's model card, in the run detail and in the
  // results view.
  modelCard: {
    button: "↓ markdown",
    title: "Download this run's model card (markdown)",
  },
  // The confidence interval a metric's tooltip explains (the run detail and the
  // results view show the same one).
  metrics: {
    ciTooltip: (percent: number, resamples: number, samples: number) =>
      `${percent}% CI (percentile bootstrap): ` +
      `${resamples} resamples of the ${samples} test ${samples === 1 ? "image" : "images"}. ` +
      `Captures the sampling uncertainty of the test split with this model ` +
      `held fixed — not the variation between training runs, which multi-seed ` +
      `replicates measure.`,
  },
  // The Colab corner of the header (ADR-163): CLI install, login per profile, balance.
  colab: {
    dialogLabel: "Google Colab",
    chipRuntimes: (n: number) => (n === 1 ? "1 runtime alive" : `${n} runtimes alive`),
    chip: {
      not_installed: "colab · off",
      installing: "colab · installing…",
      install_failed: "colab · failed",
      logged_out: "colab · sign in",
      login_pending: "colab · waiting for code",
      connected: "colab · connected",
    },
    intro:
      "Training on Google Colab uses your Google account and whatever GPU it has. VisionForge talks to Colab through the official CLI, installed in an environment of its own.",
    install: (version: string) => `install the Colab CLI (v${version})`,
    installing: "Installing the Colab CLI… this takes a minute or two.",
    installFailed: "The install failed:",
    retry: "try again",
    login: "sign in with Google",
    loginStep1: "1. Open this page, sign in with your account and authorize.",
    openPage: "open Google's page",
    loginStep2: "2. Google shows a code. Paste it here:",
    codePlaceholder: "authorization code",
    badCode: "Paste the whole code, without spaces.",
    confirm: "confirm",
    cancel: "cancel",
    account: (email: string) => `Account: ${email}`,
    accountUnknown: "Account connected (no e-mail was reported).",
    perProfile: "The sign-in applies to this profile only.",
    switchAccount: "switch account",
    balance: (units: string) => `Balance: ${units} compute units`,
    hoursLeft: (hours: number, rate: string) => `≈ ${hours} h at the current rate (${rate}/h)`,
    noRate: "No runtime is active now; the hours left depend on the GPU you pick.",
    usageFailed: "Failed to read the balance.",
    loadFailed: "Failed to reach the server.",
    trainTitle: "Train this experiment on Colab",
    trainSource: "Dataset on the VM: a zip URL, torchvision:cifar10 or hf:owner/name",
    driveSource: "Use the form's dataset folder, in Google Drive for desktop",
    driveRoot: "This computer's \"My Drive\" folder (when it is not found by itself)",
    driveRootPlaceholder: "G:\\My Drive",
    driveExport: "Also save the run to Drive (MyDrive/VisionForge/runs)",
    driveNote: "The VM mounts your Drive and copies the dataset before training; Google asks for permission on every runtime.",
    sourcePlaceholder: "https://…/dataset.zip",
    badSource: "Use an http(s) URL of a zip, torchvision:<name> or hf:<owner/name>.",
    accelerator: "GPU",
    train: "train on Colab",
    trainNote:
      "It enters the same queue as local training. The live tab follows it; the trained files stay on the VM until the import of the next phase.",
    sessionsTitle: "Runtimes alive on this account",
    sessionsNone: "No runtime is alive.",
    stopSession: "stop",
    sessionNotOurs: "started elsewhere; stop it at colab.research.google.com",
  },
  update: {
    available: (version: string) => `update available · v${version}`,
    apply: "update and reopen",
    retry: "try again",
    confirm: (version: string) =>
      `Install v${version}? The GUI closes, installs and reopens by itself; this page reloads when it is back. No training is running.`,
    confirmYes: "update now",
    confirmNo: "not now",
    skip: "skip this version",
    whatsNew: "what changed",
    updating: (version: string) => `installing v${version}… the page reloads by itself`,
    timeout:
      "The GUI is not back yet. It may be installing dependencies; reload the page in a few minutes.",
    failed: (version: string) =>
      `The update to v${version} failed and the previous version is still running.`,
    logAt: (path: string) => `Details in ${path}`,
    copy: "copy",
    copied: "copied",
    blocked: {
      source: "You are running from a source checkout. To update:",
      no_installer: "This environment has neither pip nor uv. To update:",
      remote: "Only someone at the server's own machine can update it. There, run:",
      busy: "A training is running or queued. Update once the queue is empty, or run:",
    },
  },
  header: {
    guide: "guide",
    guideTitle: "Open the guides: interface tour and first training",
    changeName: "Change name",
  },
  language: {
    label: "Language",
    switchTo: "Switch to Portuguese",
    names: { pt: "Português", en: "English" },
  },
  taskColor: {
    label: (task: string) => `${task} color`,
    title: "Task color",
    other: "Other color",
    reset: "Back to the original color",
  },
  theme: {
    label: "Theme",
    names: { system: "Automatic (follows the system)", light: "Light", dark: "Dark" },
  },
  errors: {
    cannotConnect: "Couldn't reach the server. Check that the backend is running.",
    validation: "There are validation errors in the form.",
    markdownExport: (status: number) => `Failed to export Markdown (HTTP ${status}).`,
    deviceInfo: "Failed to load devices.",
    // The preprocessing and augmentation previews fail with the same sentence.
    previewFailed: "Failed to generate the preview.",
  },
  app: {
    // The browser tab title while no run has just ended (the static one in
    // index.html is only what shows before the page loads).
    pageTitle: "VisionForge — Local AI Training",
    train: {
      simple: "▶ Train",
      cv: "▶ Run K-fold",
      sweep: "▶ Run sweep",
      replicates: "▶ Run replicates",
      compare: "▶ Compare architectures",
    },
    error: "Error",
    unnamedRun: "unnamed run",
  },
  errorBoundary: {
    title: "Something went wrong on this screen",
    body: "An unexpected error interrupted this part of the interface. Reload the page to continue.",
    reload: "Reload page",
  },
  bottomBar: {
    enqueue: "Add to queue",
    enqueued: "Queued: it starts when its turn comes. Follow it from the queue button.",
    trainLockedReview: "Open the Review step to enable Train",
    trainLockedData: "Choose the dataset in the Data step to enable Train",
    history: "history",
    datasets: "datasets",
    datasetsTitle: "Download a dataset to a local folder",
    queue: "queue",
    queueTitle:
      "Open the run in progress and see the ones waiting for the GPU — you can stop one or remove another",
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
    askName: "What's your name?",
    placeholder: "type here",
    nameLabel: "Your name",
    enter: "Enter ↵",
    // How the welcome and the header greet, by pronoun (ADR-126).
    greet: { ele: "Welcome", ela: "Welcome", neutro: "Welcome" },
    pronounLabel: "Your pronouns",
    pronounNames: { ele: "he/him", ela: "she/her", neutro: "they/them" },
  },
  // Profiles on a shared server (ADR-114): one output folder per person.
  mlflow: {
    label: "Export classification to local MLflow",
    hint: "Local storage only; requires the mlflow extra. Applies to future runs.",
    saveFailed: "Failed to save MLflow setting.",
    warning: "MLflow export incomplete",
    notInstalled: "MLflow is not installed on this computer. Install the VisionForge mlflow extra and retry the export.",
    retry: "Retry export",
    retrying: "Exporting…",
    retryFailed: "Failed to retry export.",
  },
  wandb: {
    label: "Send classification runs to Weights & Biases",
    hint: "Goes to a private project of your W&B account: the public config and each epoch's metrics. The dataset, paths, code, git and machine details stay here. Applies to future runs.",
    project: "Project",
    entity: "User or team (optional)",
    entityHint: "Empty uses the key owner's user.",
    saveTarget: "Save destination",
    saveFailed: "Failed to save the W&B setting.",
    token: "W&B API key",
    tokenHint: "Create the key in your W&B account settings. It is saved on this computer, for this profile only, and never comes back to the page.",
    configured: "Key saved in this profile.",
    saveToken: "Save key",
    removeToken: "Forget key",
    shared: "This server accepts connections from other machines: whoever picks this profile can upload with the saved key.",
    tokenFailed: "Failed to save the W&B key.",
    open: "Open in W&B",
    warning: "W&B upload incomplete",
    notInstalled: "W&B is not installed on this computer. Install the VisionForge wandb extra and retry the upload.",
    tokenMissing: "This profile has no W&B key. Save one in the profile settings and retry the upload.",
    notPrivate: "The project already exists on W&B and is not private. Pick another project name or make it private on W&B.",
    retry: "Retry W&B upload",
    retrying: "Uploading…",
    retryFailed: "Failed to retry the W&B upload.",
  },
  codeTour: {
    tab: "Code",
    intro: "This is the real code training your model, read from the installed VisionForge: not a summary. The marked parts are the path this run's settings take; the faded ones are branches it does not use.",
    sectionsLabel: "Parts of the training",
    sections: {
      data: "Data",
      model: "Model",
      loss: "Loss",
      train: "Training (one epoch)",
      validate: "Validation",
      test: "Test",
    } as Record<string, string>,
    where: (file: string, name: string, line: number) => `${file} · ${name} · line ${line}`,
    lines: (first: number, last: number) => (first === last ? `line ${first}` : `lines ${first}–${last}`),
    active: "this run's path",
    inactive: "not used in this run",
    loading: "Loading the code…",
    failed: "Failed to load the code.",
    onlyClassification: "For now, the annotated code exists for classification only.",
    perEpoch: "The training reports progress per epoch, not per line: the mark shows which path the code takes, not the line running right now.",
  },
  zenodo: {
    title: "Zenodo draft",
    open: "Create a Zenodo draft",
    intro: "Zenodo archives the experiment and, when you publish, mints a DOI to cite in the paper. Only a draft leaves from here: title, authors, license and publishing stay with you, on the Zenodo site.",
    site: "Where to create it",
    production: "Zenodo (zenodo.org)",
    sandbox: "Test sandbox (sandbox.zenodo.org)",
    sandboxHint: "The Sandbox is a test Zenodo with its own account and token: its DOIs are not real and its records may be deleted. Use it to check first.",
    token: "Zenodo token",
    tokenHint: "Create it under Account → Applications → Personal access tokens, with the deposit:write scope only. Without deposit:actions, the token cannot publish. It is saved on the server for this profile and never shown again.",
    configured: "Token configured for this profile.",
    saveToken: "Save token",
    removeToken: "Remove token",
    shared: "Shared server: profiles are not access control. Other people who reach this server can use this profile's token.",
    loading: "Building the package…",
    preview: "What goes into the draft",
    file: (name: string, size: string) => `${name} · ${size}`,
    pathsNote: "In run.json, this machine's paths keep only their final name (the dataset folder becomes just its name), without your user name.",
    leftOut: (n: number) => (n === 1 ? "1 item was left out" : `${n} items were left out`),
    checkpointWarning: "The checkpoint goes as it was saved and may contain internal metadata, paths and labels. Review where the file came from before uploading.",
    confirm: "Nothing is published: the draft is not public until you publish it on Zenodo.",
    send: "Create draft",
    busy: "Uploading…",
    failed: "Failed to complete the Zenodo operation.",
    success: "Open the draft on Zenodo",
    next: "There, fill in title, authors and license. Publishing mints a permanent DOI, and a published record cannot be deleted.",
    close: "Close",
  },
  github: {
    title: "Send to GitHub",
    open: "Send the report to GitHub",
    intro: "Keeps this run in a private experiments repository, in a folder of its own: a README with the metrics, the per-epoch table and the charts, plus run.json. The weights do not go: they belong on the Hugging Face Hub or Zenodo.",
    loading: "Building the files…",
    token: "GitHub token",
    tokenHint: "Create it under Settings → Developer settings → Fine-grained tokens, for the experiments repository only, with the Contents: read and write permission. It is saved on the server for this profile and never shown again.",
    configured: "Token configured for this profile.",
    saveToken: "Save token",
    removeToken: "Remove token",
    shared: "Shared server: profiles are not access control. Other people who reach this server can use this profile's token.",
    preview: "Files in the commit",
    folder: (folder: string) => `Folder in the repository: ${folder}`,
    noWeights: "No weights. In run.json, this machine's paths keep only their final name.",
    readme: "README preview",
    repo: "Private repository (user/repository)",
    repoPlaceholder: "user/experiments",
    confirm: "The repository must exist and be private; a public one is refused. Everything goes in one commit, on the default branch.",
    send: "Send to the repository",
    busy: "Uploading…",
    failed: "Failed to complete the GitHub operation.",
    success: "Open the folder on GitHub",
    close: "Close",
  },
  hub: {
    checkpointWarning: "The checkpoint is uploaded as saved and may contain internal metadata, paths and labels. Review the file’s origin before uploading.",
    title: "Send to Hugging Face Hub",
    open: "Review Hub upload",
    loading: "Loading preview…",
    preview: "Files to upload",
    card: "Model card preview",
    token: "Hugging Face token",
    tokenHint: "The token is saved on the server for this profile and is never displayed again.",
    configured: "Token configured for this profile.",
    saveToken: "Save token",
    removeToken: "Remove token",
    shared: "Shared server: profiles do not provide access control. Other people with access to this server can use this profile’s token.",
    repo: "Destination repository (user/model)",
    confirm: "Review the files and destination. Uploading publishes these files to the specified repository; visibility follows the repository settings.",
    send: "Upload reviewed files",
    busy: "Processing…",
    failed: "Failed to complete the Hub operation.",
    success: "Open repository on the Hub",
    close: "Close preview",
  },
  profile: {
    who: "Who are you?",
    defaultName: "Default (no profile)",
    defaultNamed: (name: string) => `${name} (default)`,
    defaultShort: "Default",
    listLabel: "Profiles",
    create: "create new profile",
    createTitle: "New profile",
    createLink: "create profile",
    nameLabel: "Profile name",
    namePlaceholder: "e.g. Ana Souza",
    folder: (slug: string) => `folder: outputs/profiles/${slug}`,
    folderNone: "Use at least one letter (a-z) or a digit.",
    submit: "Create ↵",
    creating: "Creating…",
    back: "← back",
    notice:
      "Profiles organise training history; they do not protect it. Anyone who can reach this server can pick any profile.",
    chip: "profile",
    chipTitle: "Switch profile",
    exists: "A profile with that name already exists.",
    invalid:
      "That name cannot become a folder. Use at least one letter (a-z) or a digit.",
    failed: "Couldn't create the profile.",
    queueOwner: (name: string) => `profile ${name}`,
  },
  datasetsOverlay: {
    kicker: "// datasets",
    title: "Get data",
    description:
      "Download once to a local folder, then point any task's dataset field at it.",
    tabGet: "Download",
    tabExplore: "Explore",
    tabsLabel: "Dataset tools",
  },
  datasetExplorer: {
    intro:
      "Read-only: look at a dataset before training. Nothing in the folder is changed.",
    task: "Dataset type",
    taskClassification: "Classification (class folders)",
    taskDetection: "Detection (YOLO)",
    baseDir: "Dataset folder",
    baseDirPlaceholder: "e.g. C:/datasets/coffee  or  /home/user/data",
    browse: "Browse",
    browseTitle: "Open the system's native folder picker",
    pickCancelled: "Selection canceled.",
    pickFailed: "Failed to open the folder picker.",
    open: "Open",
    refresh: "Refresh",
    loading: "Reading the folder…",
    loadFailed: (reason: string) => `Could not read the dataset: ${reason}`,
    empty: "This folder has no images in the expected layout.",
    totalImages: (n: number) => `Images: ${n}`,
    splitNames: { train: "Train", val: "Validation", test: "Test" },
    splitMissing: "folder not found",
    unlabeled: (n: number) => `without boxes: ${n}`,
    distributionTitle: "Distribution",
    imagesPerClass: "Images per class",
    boxesPerClass: "Boxes per class",
    classColumn: "Class",
    totalColumn: "Total",
    galleryTitle: "Gallery",
    filterSplit: "Split",
    filterClass: "Class",
    all: "All",
    range: (from: number, to: number, total: number) =>
      `${from}-${to} of ${total}`,
    previous: "Previous",
    next: "Next",
    noMatch: "No image matches these filters.",
    thumbBroken: "cannot open",
    boxesOnImage: (n: number) => `boxes: ${n}`,
    scanTitle: "Integrity scan",
    scanHelp:
      "Opens every image once: finds files that do not open, exact copies (same SHA-256) and the same image in different splits, and measures sizes.",
    scanLimit: "Image limit",
    scanLimitHint: "Above this, the scan reads an evenly spaced sample.",
    scanStart: "Scan",
    scanCancel: "Cancel scan",
    scanCancelling: "Canceling…",
    scanProgress: (done: number, total: number) =>
      `Scanning: ${done} of ${total}`,
    scanFailed: (reason: string) => `The scan failed: ${reason}`,
    scanBusy: "Another scan is already running.",
    scanComplete: (n: number) => `Scanned all images: ${n}`,
    scanSampled: (scanned: number, total: number) =>
      `Sample of ${scanned} out of ${total} images. A copy or leak found is real; not finding one proves nothing. Raise the limit to scan everything.`,
    scanCancelled: (scanned: number) =>
      `Canceled after ${scanned} images. The results below are partial.`,
    noProblems: "No corrupted file, copy or leak found.",
    problemsTitle: "Problems",
    corruptedTitle: "Images that do not open",
    duplicatesTitle: "Exact copies (same content)",
    leakageTitle: "Leakage between splits",
    leakageHelp:
      "The same image in more than one split makes the validation metric look better than it is.",
    groupFiles: (n: number) => `files: ${n}`,
    labelConflict: "same image, different classes",
    moreNotShown: (n: number) => `${n} more not shown`,
    sizesTitle: "Image sizes",
    longestSide: "Longest side (px)",
    aspectRatio: "Aspect ratio (width / height)",
    widthStats: (min: number, median: number, max: number) =>
      `width min ${min} · median ${median} · max ${max}`,
    heightStats: (min: number, median: number, max: number) =>
      `height min ${min} · median ${median} · max ${max}`,
    distinctSizes: (n: number) => `Distinct sizes: ${n}`,
    topSizes: "Most common sizes",
    sizeCount: (w: number, h: number, n: number) => `${w}×${h}: ${n}`,
  },
  datasetIncorporate: {
    open: "Add new images",
    close: "Close",
    intro:
      "Creates a new dataset next to the original, which does not change. Images already there stay where they are; the new ones are distributed among the splits.",
    sourceTitle: "1. Folder with the new images",
    sourceClassification: "One subfolder per class, with the images inside.",
    sourceDetection:
      "Each image with its .txt (YOLO). If the folder has classes.txt or data.yaml, classes are matched by name.",
    sourceDir: "Folder",
    sourcePlaceholder: "e.g. C:/new-images",
    browse: "Browse",
    pickCancelled: "Selection canceled.",
    pickFailed: "Failed to open the folder picker.",
    analyze: "Analyze",
    analyzing: (done: number, total: number) =>
      `Reading and comparing: ${done} of ${total}`,
    cancel: "Cancel",
    cancelling: "Canceling…",
    cancelled: "Canceled. No folder was created.",
    failed: (reason: string) => `It did not work: ${reason}`,
    previewTitle: "2. Preview",
    found: (n: number) => `Images in the folder: ${n}`,
    willAdd: (n: number) => `Will be added: ${n}`,
    willSkip: (n: number) => `Left out: ${n}`,
    nothingToAdd: "No new image passed the checks.",
    addedPerClass: "Added, per class and split",
    newClassesTitle: "New classes",
    newClassesHelp:
      "The dataset does not have these classes. They are added only if you tick them. They go at the end of the list, without renumbering the existing ones.",
    newClassItem: (name: string, n: number) => `${name} (${n} images)`,
    ratioTitle: "Proportion of the new images",
    ratioSame: (label: string) => `The dataset's own (${label})`,
    ratioCustom: "Choose",
    ratioPercent: (split: string) => `${split} (%)`,
    ratioInvalid: "Use numbers that are zero or more, with at least one above zero.",
    noSplit: (split: string) => `The dataset has no ${split}; that split will not be created.`,
    seed: "Seed",
    seedHint: "Same seed, same split. It is recorded in the new dataset.",
    seedInvalid: "Use an integer that is zero or more.",
    excludedTitle: (n: number) => `Left out (${n})`,
    reasons: {
      unreadable: "does not open",
      duplicate_existing: "exact copy of an image in the dataset",
      duplicate_new: "exact copy of another new image",
      no_label: "no .txt file",
      bad_label: "invalid .txt file",
      no_class_folder: "outside a class folder",
      unknown_class: "new class not ticked",
    } as Record<string, string>,
    reasonCount: (reason: string, n: number) => `${reason}: ${n}`,
    moreNotShown: (n: number) => `${n} more not shown`,
    destTitle: "3. Destination folder",
    destName: "Name of the new folder",
    destHint: "It is created next to the original dataset.",
    destInvalid: "Invalid folder name.",
    apply: "Create new dataset",
    applying: (done: number, total: number) => `Writing: ${done} of ${total}`,
    doneTitle: "Dataset created",
    done: (n: number, path: string) => `${n} new images in ${path}`,
    linkedCopied: (linked: number, copied: number) =>
      `${linked} files hardlinked, ${copied} copied`,
    openResult: "Open in the explorer",
  },
  datasetMerge: {
    open: "Merge with another dataset",
    close: "Close",
    intro:
      "Creates a new dataset next to the opened one (A), merging it with another (B). Neither changes. Every image stays in the split it was in.",
    sourceTitle: "1. The other dataset (B)",
    sourceHelp: "Same type and same folder layout as the opened dataset.",
    sourceDir: "Folder",
    sourcePlaceholder: "e.g. C:/datasets/other",
    browse: "Browse",
    pickCancelled: "Selection canceled.",
    pickFailed: "Failed to open the folder picker.",
    analyze: "Analyze",
    analyzing: (done: number, total: number) =>
      `Reading and comparing: ${done} of ${total}`,
    cancel: "Cancel",
    cancelling: "Canceling…",
    cancelled: "Canceled. No folder was created.",
    failed: (reason: string) => `It did not work: ${reason}`,
    found: (a: number, b: number) => `Images: A ${a} · B ${b}`,
    repeatedWithin: (a: number, b: number) =>
      `Exact copies inside one dataset (kept as they are): A ${a} · B ${b}`,
    mappingTitle: "2. Classes",
    mappingHelp:
      "Each class keeps its own name, so classes with the same name merge. Change the target to rename; two equal targets merge; tick discard to remove a class.",
    sideA: "A (opened)",
    sideB: "B (other)",
    colOrigin: "From",
    colClass: "Class",
    colImages: "Images",
    colTarget: "Becomes",
    colDiscard: "Discard",
    targetLabel: (name: string) => `Target of class ${name}`,
    discardLabel: (name: string) => `Discard class ${name}`,
    merged: "merged",
    problems: {
      empty: "Give it a name or discard the class.",
      invalid: "Invalid name for a class.",
      case: "Differs from another target only by letter case; use the same name to merge.",
    } as Record<string, string>,
    similarHint: (names: string) =>
      `Similar names that were not merged: ${names}. If they are the same class, give both the same target.`,
    resultTitle: "3. Result",
    classesTitle: "Classes of the result, in order",
    boxesColumn: "Boxes",
    splitsTitle: "Splits",
    splitCounts: (a: number, b: number) => `A ${a} · B ${b}`,
    splitOnlyIn: (split: string, side: string) =>
      `${split} exists only in ${side}: all its images come from there.`,
    totalImages: (n: number) => `Images in the result: ${n}`,
    background: (n: number) => `Images without boxes, kept: ${n}`,
    nothingLeft: "With this mapping no image is left.",
    droppedTitle: (n: number) => `Left out (${n})`,
    reasons: {
      discarded_class: "discarded class",
      all_boxes_discarded: "lost all its boxes",
    } as Record<string, string>,
    reasonCount: (reason: string, n: number) => `${reason}: ${n}`,
    droppedItem: (side: string, path: string, reason: string) =>
      `${side}: ${path} — ${reason}`,
    moreNotShown: (n: number) => `${n} more not shown`,
    duplicatesTitle: (n: number) => `Exact copies of A found in B, taken out of B (${n})`,
    duplicatesHelp: "A's copy stays. If A's copy does not survive, B's is kept.",
    leakTitle: (n: number) =>
      `${n} of them sat in different splits in the two datasets: leakage avoided.`,
    duplicateItem: (b: string, a: string) => `${b} = ${a}`,
    destTitle: "4. Destination folder",
    destName: "Name of the new folder",
    destHint: "It is created next to the opened dataset (A).",
    destInvalid: "Invalid folder name.",
    apply: "Create new dataset",
    applying: (done: number, total: number) => `Writing: ${done} of ${total}`,
    doneTitle: "Dataset created",
    done: (n: number, path: string) => `${n} images in ${path}`,
    linkedCopied: (linked: number, copied: number) =>
      `${linked} files hardlinked, ${copied} copied`,
    openResult: "Open in the explorer",
  },
  datasetResplit: {
    open: "Re-split",
    close: "Close",
    intro:
      "Creates a new dataset next to the original, which does not change. The train and validation images are pooled and redistributed by the chosen ratio; the test split stays as it is, so results remain comparable with earlier ones.",
    groupTitle: "1. Groups (optional)",
    groupHelp:
      "A CSV with the columns path,group, with path relative to the dataset folder. Images of the same group always go to the same split. Images the file does not list are a group each.",
    groupFile: "Group file",
    groupPlaceholder: "e.g. C:/data/groups.csv (empty: no groups)",
    analyze: "Analyze",
    analyzing: (done: number, total: number) => `Reading and computing: ${done} of ${total}`,
    cancel: "Cancel",
    cancelling: "Canceling…",
    cancelled: "Canceled. No folder was created.",
    failed: (reason: string) => `It did not work: ${reason}`,
    previewTitle: "2. Options and preview",
    found: (n: number) => `Images: ${n}`,
    groupsInfo: (groups: number, ungrouped: number) =>
      `${groups} ${groups === 1 ? "group" : "groups"}; ${ungrouped} ${ungrouped === 1 ? "image" : "images"} not in the file`,
    unknownEntries: (n: number) =>
      n === 1
        ? "1 row of the file is not an image of the dataset (ignored)"
        : `${n} rows of the file are not images of the dataset (ignored)`,
    lockTest: "Keep the test split as it is",
    lockNoTest: "The dataset has no test split",
    lockOnHelp: "The test split does not change; only train and validation are redistributed.",
    lockOffHelp:
      "Every split is pooled and redistributed. Results will not be comparable with runs that used the current test split.",
    byGroup: "Split by group",
    ratioCustom: "Choose the ratio",
    ratioSame: "Keeps the current proportion of the splits being redistributed.",
    ratioPercent: (split: string) => `${split} (%)`,
    ratioHelp:
      "Zero removes the split from the result. A split the dataset lacks is only created if you give it a value above zero.",
    ratioInvalid: "Use numbers greater than or equal to zero, with at least one above zero.",
    seed: "Seed",
    seedHint: "Same seed, same split. It is recorded in the new dataset.",
    seedInvalid: "Use an integer greater than or equal to zero.",
    keptLine: (n: number) => `In the result: ${n} ${n === 1 ? "image" : "images"}`,
    movedLine: (n: number) => `Change split: ${n}`,
    droppedLine: (n: number) => `Left out: ${n}`,
    tableTitle: "Images per class and split (before → after)",
    lockedMark: "(locked)",
    ratiosTitle: "Requested and achieved ratio",
    ratioLine: (split: string, requested: string, achieved: string) =>
      `${split}: requested ${requested}%, achieved ${achieved}%`,
    duplicatesLine: (sets: number, extras: number) =>
      `Exact copies kept together: ${sets} ${sets === 1 ? "set" : "sets"} (${extras} extra ${extras === 1 ? "copy" : "copies"}).`,
    duplicatesLeak: (n: number) =>
      n === 1
        ? "One of them sat in different splits: leakage avoided."
        : `${n} of them sat in different splits: leakage avoided.`,
    warnNewSplit: (split: string) => `${split} did not exist and will be created.`,
    warnEmptySplit: (split: string) => `${split} would get no images.`,
    warnSplitRemoved: (split: string) => `${split} no longer exists in the result.`,
    warnSplitEmptyResult: (split: string) => `Warning: ${split} ends up with no images.`,
    warnClassMissing: (cls: string, split: string) =>
      `Class ${cls} has no images in ${split}.`,
    droppedTitle: (n: number) => `Left out (${n})`,
    droppedHelp:
      "They would leak into training: they are copies of locked test images or belong to a group that has an image in the test split.",
    reasons: {
      duplicate_of_test: "exact copy of a test image",
      group_in_test: "group with an image in the test split",
    } as Record<string, string>,
    reasonCount: (reason: string, n: number) => `${reason}: ${n}`,
    moreNotShown: (n: number) => `${n} more not shown`,
    destTitle: "3. Destination folder",
    destName: "Name of the new folder",
    destHint: "It is created next to the original dataset.",
    destInvalid: "Invalid folder name.",
    apply: "Create new dataset",
    applying: (done: number, total: number) => `Writing: ${done} of ${total}`,
    doneTitle: "Dataset created",
    done: (n: number, moved: number, path: string) =>
      `${n} ${n === 1 ? "image" : "images"} in ${path}; ${moved} changed split`,
    linkedCopied: (linked: number, copied: number) =>
      `${linked} ${linked === 1 ? "file" : "files"} hardlinked, ${copied} copied`,
    openResult: "Open in the explorer",
  },
  datasetCoco: {
    exportOpen: "Export to COCO",
    importOpen: "Import COCO",
    close: "Close",
    exportIntro:
      "Creates a new COCO dataset next to the original, which does not change: images in images/<split> and one annotations/instances_<split>.json per split. The explorer does not read COCO, so the result is not opened here.",
    importIntro:
      "Reads a COCO folder (annotation .json files and their images) and creates a new YOLO dataset next to it. The folder is only read. Only the boxes are imported; segmentation polygons are ignored.",
    analyze: "Analyze",
    analyzingExport: (done: number, total: number) =>
      `Reading labels and image sizes: ${done} of ${total}`,
    analyzingImport: (done: number, total: number) =>
      `Reading images and converting boxes: ${done} of ${total}`,
    applying: (done: number, total: number) => `Writing: ${done} of ${total}`,
    cancel: "Cancel",
    cancelling: "Canceling…",
    cancelled: "Canceled. No folder was created.",
    failed: (reason: string) => `It did not work: ${reason}`,
    folder: "COCO folder",
    folderPlaceholder: "e.g. C:/datasets/coco-mini",
    browse: "Browse",
    browseTitle: "Open the system's native folder picker",
    pickCancelled: "Selection canceled.",
    pickFailed: "Failed to open the folder picker.",
    previewTitle: "2. Preview",
    exportTitle: "1. Analyze the opened dataset",
    importTitle: "1. COCO folder",
    imagesLine: (n: number) => `Images: ${n}`,
    boxesLine: (n: number) => `Boxes: ${n}`,
    categoriesLine: (n: number) => `${n} ${n === 1 ? "category" : "categories"}`,
    splitColumn: "Split",
    imagesColumn: "Images",
    boxesColumn: "Boxes",
    excludedTitle: (n: number) => `Left out (${n})`,
    excludedHelp:
      "These images are not in the result. An image is never half-converted: it is either complete or left out.",
    exportReasons: {
      unreadable_image: "the image does not open",
      malformed_label: "malformed line in the label file",
    } as Record<string, string>,
    reasonCount: (reason: string, n: number) => `${reason}: ${n}`,
    moreNotShown: (n: number) => `${n} more not shown`,
    jsonFile: (name: string) => `File ${name}`,
    splitFor: (name: string) => `Split of ${name}`,
    splitUnset: "Choose…",
    splitSkip: "Skip this file",
    splitNames: { train: "Train", val: "Validation", test: "Test" } as Record<string, string>,
    detectedHint: (split: string) => `Detected from the file name: ${split}`,
    undetectedHint: "The file name does not say the split: choose one or skip the file.",
    imagesFolder: (dir: string) => `Images in: ${dir === "." ? "the folder itself" : dir}`,
    imagesFolderMissing: "No folder with the images was found.",
    sourceCounts: (listed: number, kept: number, annotations: number, boxes: number) =>
      `${listed} ${listed === 1 ? "image" : "images"} listed, ${kept} kept; ${annotations} ${annotations === 1 ? "annotation" : "annotations"}, ${boxes} ${boxes === 1 ? "box" : "boxes"}`,
    skippedTitle: "Left out",
    adjustedTitle: "Kept, but changed",
    importReasons: {
      missing_image: "image not found on disk",
      unreadable_image: "the image does not open",
      unsafe_path: "file_name points outside the folder",
      bad_image_entry: "malformed image entry",
      duplicate_image_id: "repeated image id",
      crowd: "iscrowd box (skipped)",
      bad_bbox: "malformed bbox",
      degenerate_box: "no area left after clipping to the image",
      unknown_image_id: "annotation of an image that is not listed",
      unknown_category: "annotation of a category that is not listed",
      bad_category: "malformed category entry",
      size_mismatch: "the real image size differs from the JSON (the real one is used)",
      clipped_box: "box clipped to the image",
    } as Record<string, string>,
    ignoredTitle: "JSON files that are not COCO annotations (ignored)",
    ignoredReasons: {
      unreadable: "cannot be read",
      not_json: "not valid JSON",
      not_coco: "no images, annotations and categories lists",
    } as Record<string, string>,
    splitConflict: (split: string) =>
      `More than one file is assigned to ${split}. Change one of them or skip it.`,
    unassigned: (n: number) =>
      n === 1
        ? "1 file still needs a split (or to be skipped)."
        : `${n} files still need a split (or to be skipped).`,
    planTitle: "3. Result",
    planLine: (images: number, boxes: number) =>
      `In the result: ${images} ${images === 1 ? "image" : "images"}, ${boxes} ${boxes === 1 ? "box" : "boxes"}`,
    mappingTitle: "Classes (COCO id → YOLO id)",
    mappingLine: (cocoId: number, yoloId: number, name: string) =>
      `${cocoId} → ${yoloId}  ${name}`,
    renamedMark: (original: string) => `(was "${original}"; the name was already taken)`,
    destTitle: "Destination folder",
    destName: "Name of the new folder",
    destHint: "It is created next to the original folder.",
    destInvalid: "Invalid folder name.",
    applyExport: "Create COCO dataset",
    applyImport: "Create YOLO dataset",
    doneTitle: "Dataset created",
    doneExport: (images: number, boxes: number, excluded: number, path: string) =>
      `${images} ${images === 1 ? "image" : "images"} and ${boxes} ${boxes === 1 ? "box" : "boxes"} in ${path}; ${excluded} left out`,
    doneImport: (images: number, boxes: number, classes: number, path: string) =>
      `${images} ${images === 1 ? "image" : "images"}, ${boxes} ${boxes === 1 ? "box" : "boxes"} and ${classes} ${classes === 1 ? "class" : "classes"} in ${path}`,
    linkedCopied: (linked: number, copied: number) =>
      `${linked} ${linked === 1 ? "file" : "files"} hardlinked, ${copied} copied`,
    openResult: "Open in the explorer",
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
    inProgress: "Training in progress…",
    runId: "Run ID:",
  },
  advancedFields: {
    advanced: "advanced",
    changed: "values changed",
  },
  modelAdvice: {
    suggested: (architecture: string, optimizer: string, learningRate: number) =>
      `For ${architecture}, we suggest ${optimizer} at ${learningRate}.`,
    apply: (optimizer: string, learningRate: number) =>
      `use ${optimizer} · ${learningRate}`,
    // Only what was measured is said (ADR-099/100): one model per family, on 4
    // classes. When `measuredOn` is not `architecture`, the note says so rather
    // than letting a sibling borrow the number.
    collapseMeasured: (facts: CollapseFacts) => measuredNote(facts, "predicted a single class"),
    failsToLearnMeasured: (facts: CollapseFacts) => measuredNote(facts, "learned little"),
    unmeasured: (architecture: string, optimizer: string, learningRate: number) =>
      `${architecture}: Adam at 1e-3 was not measured for this family; we suggest ${optimizer} at ${learningRate}, the same as the measured attention families.`,
    upscaling: (medianSide: number) =>
      `The images are about ${medianSide}px on a side; training above that upsizes the image without adding detail.`,
  },
  infoDot: {
    label: "Explanation",
  },
  textField: {
    showSecret: "Show what you typed",
    hideSecret: "Hide what you typed",
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
      "How many times the model sees the whole dataset. More epochs mean more learning, until the model starts memorizing the data.",
    batch_size:
      "How many images per step. A larger batch gives steadier gradients and uses more VRAM; if you run out of memory, lower this one first.",
    learning_rate:
      "The size of each update step. Too high and it diverges, too low and it never gets there.",
    seed:
      "Pins down the random draws (initial weights, data order). The same seed on the same data gives the same result.",

    // ── advanced: optimization ────────────────────────────────────────────────
    optimizer:
      "The algorithm that turns gradients into weight updates. Adam converges fast with little tuning; SGD usually generalizes better if you give it time.",
    momentum:
      "How much the previous step carries into the current one. Smooths the trajectory and helps cross plateaus.",
    weight_decay:
      "Pulls the weights toward zero. Fights overfitting; too high and the model can't learn.",
    learning_rate_final: lrFinalHelp,
    lrf: lrFinalHelp,

    // ── advanced: scheduling ──────────────────────────────────────────────────
    scheduler:
      "How the learning rate decreases during training. Letting it decay almost always helps.",
    step_size: "How many epochs between learning rate reductions.",
    gamma: "The factor the learning rate is multiplied by at each reduction.",
    cos_lr: "Makes the learning rate follow a cosine curve instead of decaying linearly.",
    warmup_epochs:
      "Epochs at the start during which the learning rate ramps up gradually, so early training stays stable.",

    // ── advanced: stopping and regularization ─────────────────────────────────
    early_stopping_patience:
      "Consecutive epochs without improvement before training stops. Set to 0 (or leave empty) to run every configured epoch.",
    patience: "Epochs without improvement before training stops early.",
    // The scheduler's own `patience` (reduce on plateau) is a different knob from
    // early stopping; ParamPanel looks the dot-path up before the leaf name.
    "training.scheduler.patience":
      "Epochs without improvement before the learning rate is reduced.",
    label_smoothing: "Softens the labels so the model doesn't become overconfident.",
    dropout:
      "Randomly switches off neurons during training, so the model can't rely on just a few.",
    freeze: "Freezes the first N layers. Useful for transfer learning with little data.",

    // ── advanced: mechanics ───────────────────────────────────────────────────
    amp:
      "Mixed precision: computes in 16-bit where it can. Trains faster and uses less VRAM, with a low risk of instability.",
    mixed_precision:
      "Runs part of the math in 16-bit precision. Speeds up training and uses less VRAM on recent GPUs; on sensitive models it can cost numerical precision.",
    live_gradcam_every:
      "How often, in epochs, to show in the Results tab where the model looks on a few fixed validation images (Grad-CAM). Also shown on the last epoch and on every new best. 0 turns it off. It does not change the training: with the same seed, the numbers come out the same.",
    deterministic:
      "Makes the same config with the same seed return exactly the same numbers. On by default: we measured the cost, and it is zero, or even negative, on short runs.",
    base_dir:
      "Root folder of the dataset. Inside it are the train, validation and test subfolders — VisionForge looks for the usual names (train/val/test, treino/validacao/teste) and fills them in automatically when it finds them. What each subfolder holds depends on the task: one folder per class for classification, images and labels for detection, images and masks for segmentation.",
    train_dir:
      "Subfolder used to fit the weights. It's the only one the model learns from.",
    val_dir:
      "Subfolder used every epoch to measure progress and pick the best checkpoint. It plays no part in fitting the weights.",
    test_dir:
      "Subfolder evaluated only once, at the end, so the result you report hasn't influenced any decision.",
    coreset_ratio:
      "How much of the \"normal\" data PatchCore keeps to compare against later. It cuts the training images into small patches and keeps a sample of them, as varied as possible; a new image is anomalous when one of its patches looks unlike anything kept. 1% is the value from the original paper. Raising it makes the memory bank more complete, but build time grows proportionally: 10% takes ten times as long.",
    num_workers: workersHelp,
    workers: workersHelp,
    pin_memory: "Speeds up copying the images to the GPU. Leave it on unless you're short on RAM.",
    image_size:
      "Training resolution. Larger captures more detail, but VRAM and time grow with the square of the size.",
    nbs: "Nominal batch size used to normalize weight decay when the actual batch is smaller.",
    single_cls:
      "Treats all classes as one. Use it to measure only how well the boxes are localized.",
    rect:
      "Groups images with similar aspect ratios instead of forcing squares. Faster, less uniform.",
    multi_scale:
      "Varies the resolution between steps, so the model copes with objects of different sizes.",
    close_mosaic:
      "Turns off mosaic augmentation for the last N epochs, so the model finishes training on regular, unstitched images.",
    box: "Weight of the box localization loss.",
    cls: "Weight of the classification loss.",
    dfl: "Weight of the distribution focal loss (DFL), which refines the box edges.",
  },
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
      weights_path: "Custom checkpoint (.pth)",
      learning_rate: "Learning rate",
      epochs: "Epochs",
      batch_size: "Batch size",
      early_stopping_patience: "Early stop (patience)",
      optimizer: "Optimizer",
      weight_decay: "Weight decay",
      seed: "Seed",
      deterministic: "Deterministic",
      mixed_precision: "Mixed precision (AMP)",
      live_gradcam_every: "Live Grad-CAM (every N epochs)",
      kind: "Type",
      step_size: "Step size",
      gamma: "Gamma",
      patience: "Patience",
      factor: "Factor",
      min_lr: "Min LR",
      base_dir: "Base folder",
      train_dir: "Train subfolder",
      val_dir: "Validation subfolder",
      test_dir: "Test subfolder",
      num_workers: "Workers",
      pin_memory: "Pin memory",
      image_size: "Image size",
      horizontal_flip: "Horizontal flip",
      rotation_degrees: "Rotation (degrees)",
      color_jitter: "Color jitter",
      normalize_mean: "Normalization (mean)",
      normalize_std: "Normalization (std)",
      n_folds: "Folds",
      stratified: "Stratified",
      shuffle: "Shuffle",
      fold_seed: "Fold seed",
      mode: "Mode",
      unfreeze_from_layer: "Unfreeze from layer",
      backbone_lr_multiplier: "Backbone LR (×)",
      model_names: "Architectures",
      metric: "Ranking metric",
      lr_policy: "Learning rate per model",
    },
    kickers: {
      strategy: "// experiment strategy",
      scheduler: "// learning rate scheduler",
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
      simple: "Single run",
      crossValidation: "K-fold (CV)",
      transferLearning: "Transfer learning",
      gridSearch: "Grid search",
      randomSearch: "Random search",
    },
    blockHints: {
      crossValidation:
        "Trains N models, one per fold of the training folder, each validated on its own held-out fold. `normalize_mean/std` are recomputed per fold to avoid data leakage. The test split is not used; results are aggregated in `cv_summary.json`.",
      transferLearning:
        "Feature extraction (trains only the head) or fine-tuning (head + part of the backbone with a smaller LR). Useful for small datasets, since it keeps the pretrained features intact. In feature extraction the backbone weights don't move, but the BatchNorm statistics recalibrate to your dataset — the backbone is frozen, but not untouched.",
      gridSearch:
        'Trains **once per combination** of the Cartesian product of the space defined below. Each key is a dot-path (e.g. `training.learning_rate`); the value is a list. Careful: 3×3×2 is already 18 trials. To **compare architectures**, add values to the "Architecture" field ("+ add to grid" button) — a single-axis grid — then compare the runs in history.',
      randomSearch:
        "Samples `n_trials` independent configurations from the space below. Each parameter has a type: `uniform`, `log_uniform` (learning rate, weight decay) or `choice` (discrete lists).",
    },
    weights: {
      label: "Custom checkpoint (.pth)",
      clearTitle: "Remove the custom checkpoint (falls back to pretrained / random weights)",
      clear: "clear",
      placeholder: "optional — replaces the ImageNet weights",
      browse: "Browse",
      cancelled: "Canceled.",
      pickFailed: "Failed to open the file picker.",
    },
    grid: {
      addValue: "+ add to grid",
      addAnother: "+ add value",
      removeValue: "Remove value",
      axisTag: (values: number) => `grid · ${values} ${values === 1 ? "value" : "values"}`,
      banner:
        "**Grid search is on.** Click `+ add to grid` on any __Model__ or __Training__ hyperparameter to sweep several values.",
      lrMultiplier: "Read the learning rate axis as a multiplier of each model's suggested rate",
      lrMultiplierHint:
        "(the axis values become factors, such as 0.3, 1 and 3; the Review shows each trial's rate)",
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
    importWarnings: (count: number, summary: string, extra: number) =>
      `YAML imported with ${count} structural ${count === 1 ? "warning" : "warnings"}:\n${summary}${extra > 0 ? `\n…(+${extra} more)` : ""}`,
    importDropped: (count: number) =>
      count === 1
        ? "1 value of the wrong type was ignored; the field uses its default."
        : `${count} values of the wrong type were ignored; the fields use their defaults.`,
    importMissing: (count: number) =>
      count === 1
        ? "1 required field has no value: fill it in before training, or the backend will reject the config."
        : `${count} required fields have no value: fill them in before training, or the backend will reject the config.`,
    unavailable: (task: string) => `${task} is not available yet`,
    unavailableBody: "This task will be implemented in an upcoming phase of VisionForge.",
    loadingSchema: "loading schema…",
    hiddenParams: (n: number) =>
      `${n} hidden ${n === 1 ? "parameter" : "parameters"} — turn on to adjust ${n === 1 ? "it" : "them"}`,
    fieldErrors: (n: number) => `${n} ${n === 1 ? "field" : "fields"} with errors:`,
  },
  runDetail: {
    back: "← history",
    loading: "loading…",
    loadFailed: "Failed to load run details.",
    browse: "📁 Browse",
    copy: "copy",
    copyPath: "Copy path",
    reveal: {
      button: "Open folder",
      title: "Open the run's folder in the file manager",
      opened: "Folder opened in the file manager.",
      forbidden:
        "The folder only opens from the machine running the server. Copy the path.",
      notFound: "This run no longer exists on disk.",
      failed: "Failed to open the folder. Copy the path.",
    },
    imageFolder: "Image folder",
    resume: {
      button: "▶ resume",
      title: (done: number, total: number) =>
        `Resume from epoch ${done} up to ${total} in the same folder`,
      titleNoTotal: "Resume this run in the same folder",
      queuing: "Queuing the run to resume…",
      running: "Resuming this run — follow it in the training panel.",
      queued: "Queued: starts when the current run finishes.",
      failed: "Failed to resume.",
    },
    picker: {
      opening: "Opening the folder picker…",
      cancelled: "Canceled.",
      picked: (path: string) => `Folder: ${path}`,
      failed: "Failed to open the folder picker.",
    },
    dataset: {
      title: "Dataset",
      name: "Name",
      path: "Path",
      contents: "Contents",
      files: (n: number | null | undefined, size: string) =>
        `${n ?? "—"} ${n === 1 ? "file" : "files"} · ${size}`,
      fingerprint: "Fingerprint",
      noFingerprint: "no fingerprint — run predates 2026-07-26",
    },
    location: {
      title: "Location on disk",
      runFolder: "Run folder",
      checkpoint: "Checkpoint",
      deviceUsed: "Device used",
      env: (key: string) => `env · ${key}`,
    },
    training: {
      title: "Training configuration",
      transferLearning: "transfer learning",
    },
    pipeline: {
      title: "Applied pipeline (preprocessing + augmentation)",
      preprocessing: "// preprocessing (in order)",
      augmentation: "// augmentation & normalization",
    },
    onnx: {
      title: "Export to ONNX",
      open: "↗ export onnx",
      hint: "Converts the checkpoint to ONNX, checks its outputs against PyTorch for numerical differences and measures inference latency. The file is saved next to the checkpoint as",
      file: "best_model.onnx",
      opsetVersion: "opset_version",
      benchmarkRuns: "benchmark_runs",
      dynamicAxes: "dynamic_axes",
      validate: "validate",
      benchmark: "benchmark",
      run: "▶ Run export",
      running: "Exporting…",
      exporting: "Exporting to ONNX…",
      saved: (path: string) => `ONNX saved to ${path}`,
      failed: "Failed to export ONNX.",
      stats: {
        fileSize: "file_size",
        maxDiff: "max_diff",
        onnxLatency: "onnx latency μ",
        onnxP95: "onnx p95",
        torchLatency: "torch latency μ",
        speedup: "speedup (torch/onnx)",
        runs: "n_runs",
      },
    },
    batch: {
      title: "Batch inference (CSV)",
      open: "+ batch inference",
      folderPlaceholder: "e.g. C:/datasets/inbox",
      recursive: "recursive (subfolders)",
      run: "▶ Run inference",
      running: "Processing…",
      hint: "Runs the checkpoint over a folder of images and writes a CSV with one row per image and the model's output for the task (class and probabilities, predicted values or anomaly score and flag). Useful for processing batches of new data without retraining.",
      needFolder: "Enter the image folder for inference.",
      starting: "Running batch inference…",
      done: (ok: number, csv: string) =>
        `${ok} ${ok === 1 ? "image" : "images"} processed · CSV at ${csv}`,
      doneWithFailures: (ok: number, failed: number, csv: string) =>
        `${ok} ok · ${failed} failed · CSV at ${csv}`,
      failed: "Failed to run batch inference.",
      processed: "processed",
      failedCount: "failed",
      csv: "csv",
      failedFiles: (n: number) =>
        `${n} ${n === 1 ? "file" : "files"} failed (click to see up to 5)`,
      more: (n: number) => `…+${n} more`,
    },
    gradcam: {
      title: "Grad-CAM (explainability)",
      open: "🔥 Grad-CAM",
      folderPlaceholder: "e.g. C:/datasets/samples",
      samples: "Number of samples (1–64)",
      run: "▶ Generate Grad-CAM",
      running: "Generating…",
      hint: "Generates Grad-CAM heatmaps over sample images, highlighting the regions that most influenced the model's output: the predicted class for classification, the first target for regression and class 0 for segmentation. Useful for interpreting what the model learned.",
      needFolder: "Enter the image folder.",
      starting: "Generating Grad-CAM maps…",
      done: (count: number, layer: string) =>
        `${count} ${count === 1 ? "map" : "maps"} generated · target layer: ${layer}`,
      failed: "Failed to generate Grad-CAM.",
      classNumber: (n: number) => `class ${n}`,
      predictedIs: (label: string) => `predicted: ${label}`,
      actual: "actual",
      predicted: "predicted",
    },
    metrics: {
      title: "Metrics",
      none: "No metrics recorded.",
      labels: {
        accuracy: "Accuracy",
        precision: "Precision",
        best_val_loss: "Best val loss",
        best_epoch: "Best epoch",
        total_epochs: "Epochs trained",
      },
      onTestSet: (name: string) => `${name} (test)`,
    },
    graphs: {
      title: "Plots (click to expand)",
    },
    tests: {
      title: "Tests on this model",
      open: "+ test",
      empty: "No tests run on this model yet.",
      folder: "Test folder",
      manifest: "Test manifest (.csv)",
      folderPlaceholder: "e.g. C:/datasets/coffee_v2/test",
      manifestPlaceholder: "e.g. C:/datasets/age/test.csv",
      folderHint: {
        detection: "A folder in YOLO layout: images and their matching .txt label files.",
        segmentation:
          "A folder with an images subfolder and a masks subfolder, paired by file name.",
        anomaly:
          "A folder with the normal-images subfolder plus one or more defect subfolders, as in training.",
        regression:
          "The .csv with the image column and the target column(s). Image paths are resolved against the images subfolder next to the .csv, as in training.",
        classification: "A folder with one subfolder per class — the same convention as training.",
      },
      label: "Name (optional)",
      labelPlaceholder: "e.g. holdout_2026",
      run: "▶ Run test",
      running: "Evaluating…",
      needFolder: "Enter the test folder or .csv manifest.",
      starting: "Evaluating the model on the new dataset…",
      recorded: (id: string) => `Test recorded: ${id}`,
      failed: "Failed to run the test.",
    },
    cv: {
      title: (ok: number, total: number, failed: number, stopped: number) =>
        `Cross-validation · ${ok}/${total} folds ok${failed > 0 ? ` · ${failed} failed` : ""}${stopped > 0 ? ` · ${stopped} stopped` : ""}`,
      stdNote: (ddof: 0 | 1): string =>
        ddof === 1
          ? "sample std (divides by n−1)"
          : "population std (divides by n) — older runs; newer ones use n−1, and the two are not comparable",
      meanAccuracy: "Mean accuracy ± std",
      meanF1: "Mean F1 ± std",
      fold: "Fold",
      train: "train",
      val: "val",
      valLoss: "val_loss",
      accuracy: "accuracy",
      status: "status",
    },
  },
  detectionPanel: {
    namePlaceholder: "detection_001",
    model: {
      title: "Model · detection",
      backend: "Backend",
      backendHint: "where the model comes from",
      architecture: "Architecture",
      architectureHint: "detector",
      pretrained: "Pretrained weights",
      pretrainedHint: "COCO",
    },
    training: {
      title: "Training",
      epochs: "Epochs",
      batchSize: "Batch size",
      batchSizeHint: "any integer",
      learningRate: "Learning rate",
      learningRateHint: "lr0",
      seed: "Seed",
      patience: "Patience",
      patienceHint: "early stop",
      deterministic: "Deterministic",
      deterministicHint: "reproducible",
      optimizer: "Optimizer",
      optimizerHint: "auto = Ultralytics picks",
      momentum: "Momentum",
      momentumHint: "SGD momentum / Adam β1",
      weightDecay: "Weight decay",
      weightDecayHint: "L2",
    },
    schedule: {
      title: "LR schedule & loss weights",
      lrf: "lrf",
      lrfHint: "final LR = lr0 × lrf",
      cosLr: "Cosine LR",
      cosLrHint: "cosine schedule",
      warmupEpochs: "Warmup epochs",
      warmupMomentum: "Warmup momentum",
      warmupBiasLr: "Warmup bias LR",
      boxGain: "Box loss gain",
      boxHint: "box",
      clsGain: "Cls loss gain",
      clsHint: "cls",
      dflGain: "DFL loss gain",
      dflHint: "dfl",
    },
    mechanics: {
      title: "Regularization & mechanics",
      labelSmoothing: "Label smoothing",
      dropout: "Dropout",
      nbs: "Nominal batch size (nbs)",
      freeze: "Freeze layers",
      freezeHint: "0 = none",
      closeMosaic: "Close mosaic",
      closeMosaicHint: "turns off mosaic for the last N epochs",
      amp: "AMP",
      ampHint: "mixed precision",
      singleCls: "Single class",
      singleClsHint: "treats all classes as one",
      rect: "Rect",
      rectHint: "rectangular batches",
      multiScale: "Multi-scale",
      multiScaleHint: "varies imgsz by ±50%",
    },
    dataset: {
      title: "Dataset",
      source: "Dataset source",
      folderOption: "YOLO folder",
      yamlOption: "data.yaml",
      folderHint: "builds the data.yaml from the folder",
      yamlHint: "uses an existing data.yaml",
      baseDir: "Base folder",
      baseDirPlaceholder: "…/dataset (images/train, images/val)",
      baseDirHint: "YOLO dataset root",
      browseFolder: "📁 Browse",
      yamlFile: "data.yaml file",
      yamlPlaceholder: "…/dataset/data.yaml",
      yamlFileHint: "Ultralytics / Roboflow",
      browseYaml: "📄 Browse",
      imageSize: "Image size",
      imageSizeHint: "imgsz",
      numClasses: "Number of classes",
      numClassesHint: "filled in from the dataset below; editable",
      yamlNote:
        "The data.yaml defines the splits and class names. Check that `nc` matches the number of classes above.",
      preprocessingNote:
        "Filters are applied once and saved to a temporary copy of the dataset, used only during training and deleted afterwards. Labels are carried over unchanged.",
    },
    augmentation: {
      title: "Data augmentation",
      toggle: "Augmentation",
      toggleHint: "when off, the values below are saved for later",
      hsvHue: "HSV — hue",
      hsvSaturation: "HSV — saturation",
      hsvValue: "HSV — value",
      degrees: "Degrees",
      translate: "Translate",
      scale: "Scale",
      shear: "Shear",
      perspective: "Perspective",
      flipUpDown: "Flip up-down",
      flipLeftRight: "Flip left-right",
      bgrSwap: "BGR swap",
      mosaic: "Mosaic",
      mixup: "Mixup",
      copyPaste: "Copy-paste",
      autoAugment: "Auto augment",
      randomErasing: "Random erasing",
      probability: "probability",
      disabled: "Disabled",
    },
  },
  plots: {
    labels: {
      "loss.png": "Loss (train + val)",
      "accuracy.png": "Accuracy (train + val)",
      "confusion_matrix.png": "Confusion matrix",
      "confusion_matrix_normalized.png": "Confusion matrix (normalized)",
      "roc_curve.png": "ROC curve",
      "precision_recall_curve.png": "Precision-Recall curve",
      "results.png": "Training curves (loss + mAP)",
      "BoxPR_curve.png": "Precision-Recall curve (box)",
      "BoxF1_curve.png": "F1 curve (box)",
      "auroc.png": "AUROC over epochs",
      "BoxP_curve.png": "Precision curve (box)",
      "BoxR_curve.png": "Recall curve (box)",
      "val_batch0_pred.jpg": "Validation batch predictions",
      "pred_vs_true.png": "Predicted vs actual",
      "residuals.png": "Residual distribution",
      "iou_per_class.png": "IoU per class",
      "score_histogram.png": "Scores: normal vs defect",
    },
  },
  resultsView: {
    title: "// results",
    metricLabels: {
      best_val_loss: "Best val loss",
      best_epoch: "Best epoch",
      total_epochs: "Epochs trained",
      test_accuracy: "Accuracy",
      test_f1: "F1",
      test_precision: "Precision",
      test_recall: "Recall",
      test_auc_roc: "AUC-ROC",
      map50: "mAP@50",
      map50_95: "mAP@50-95",
      precision: "Precision (box)",
      recall: "Recall (box)",
      box_loss: "Box loss (val)",
    },
    graphsTitle: "// plots · click to expand",
    reportTitle: "// report",
    cols: {
      rank: "Rank",
      architecture: "Architecture",
      time: "time (s)",
      status: "status",
      seed: "seed",
      overrides: "overrides",
    },
    notRun: (n: number): string => ` · ${n} not run`,
    outcome: {
      ok: "ok",
      failed: (error: string) => `failed · ${error}`,
      stopped: "stopped · left out of the aggregate",
    },
    cv: {
      title: (ok: number, total: number, failed: number, stopped: number) =>
        `// k-fold cross-validation · ${ok}/${total} folds ok${failed > 0 ? ` · ${failed} failed` : ""}${stopped > 0 ? ` · ${stopped} stopped` : ""}`,
      accuracyMeanStd: "Accuracy (mean ± std)",
      f1MeanStd: "F1 (mean ± std)",
      fold: "Fold",
      trainSize: "train_size",
      valSize: "val_size",
      valLoss: "val_loss",
      accuracy: "accuracy",
    },
    taskCv: {
      title: (ok: number, total: number, metric: string, stopped: number) =>
        `// k-fold · ${ok}/${total} folds ok${stopped > 0 ? ` · ${stopped} stopped` : ""} · headline metric: ${metric}`,
      meanStd: "mean ± std across folds",
      sample: (n: number) => `n=${n}`,
      fold: "fold",
      trainVal: "train/val",
    },
    replicates: {
      title: (ok: number, total: number, metric: string, stopped: number) =>
        `// multi-seed replicates · ${ok}/${total} seeds ok${stopped > 0 ? ` · ${stopped} stopped` : ""} · headline metric: ${metric}`,
      citable: "🎯 citable result",
      headlineMeta: (hasCi: boolean, n: number) => `${hasCi ? "95% CI · " : ""}n=${n}`,
      metric: "metric",
      n: "n",
      mean: "mean",
      std: "std",
      min: "min",
      max: "max",
      ci: "95% CI",
    },
    comparison: {
      title: (ok: number, total: number, failed: number, stopped: number, metric: string) =>
        `// architecture comparison · ${ok}/${total} ok${failed > 0 ? ` · ${failed} failed` : ""}${stopped > 0 ? ` · ${stopped} stopped` : ""} · ranked by ${metric}`,
      testSetNote: "Test-set metrics. A comparison report, not a model selection.",
    },
    sweep: {
      title: (mode: string, ok: number, total: number, stopped: number, metric: string) =>
        `// ${mode} sweep · ${ok}/${total} trials ok${stopped > 0 ? ` · ${stopped} stopped` : ""} · ranked by validation ${metric}`,
      best: (metric: string) => `👑 best trial · validation ${metric}=`,
    },
    modelComparison: {
      title: (ok: number, total: number, failed: number, stopped: number) =>
        `// model comparison · ${ok}/${total} ok${failed > 0 ? ` · ${failed} failed` : ""}${stopped > 0 ? ` · ${stopped} stopped` : ""}`,
      learningRate: "Effective LR",
      accuracy: "Accuracy",
      aucRoc: "AUC-ROC",
      footer: "Top 3 above. The full ranking is in `outputs/reports/<experiment>/ranking.csv`.",
    },
    gridSearch: {
      title: (ok: number, total: number, index: string) =>
        `// grid search · ${ok}/${total} trials ok · 👑 best trial #${index}`,
      overrides: "// winning trial's overrides",
      footer:
        "Full table in `outputs/reports/<experiment>/grid_search_summary.csv` · winning config in `best_config.yaml`.",
    },
  },
  history: {
    eyebrow: "// training history",
    title: "Recent runs",
    select: "✓ Select",
    cancelSelect: "Cancel selection",
    compare: (n: number) => `↔ Compare ${n}`,
    deleteSelected: (n: number) => `🗑 Delete ${n}`,
    deleteSelectedTitle: (n: number) =>
      `Permanently delete ${n} ${n === 1 ? "run" : "runs"}`,
    loading: "loading history…",
    errorTitle: "Error",
    loadFailed: "Failed to load run history.",
    emptyTitle: "No runs yet",
    emptyHint: "Run your first experiment to see it here.",
    selectModeTip: "Selection mode — select the runs you want to delete or compare.",
    selectedTip: (n: number) =>
      `${n} selected — 🗑 deletes; ↔ compares (needs at least 2).`,
    searchPlaceholder: "🔍 search by name, architecture or run_id…",
    clearSearch: "Clear search",
    noMatch: "No runs match the current filter.",
    allTab: "All",
    allChip: "all",
    filterType: "type",
    filterBlock: "block",
    filterStatus: "status",
    sortLabel: "sort by",
    sort: {
      recent: "newest",
      oldest: "oldest",
      epochs: "most epochs",
    },
    card: {
      deleteTitle: "Permanently delete this run",
      preprocessing: (n: number) => `⚗ ${n} ${n === 1 ? "filter" : "filters"}`,
      preprocessingTitle: (n: number) =>
        `${n} preprocessing ${n === 1 ? "filter" : "filters"} applied to training`,
      resumeTitle: (done: number, total: number) =>
        `Stopped at epoch ${done} of ${total} — can be resumed`,
      resumeTitleNoTotal: "Stopped before the end — can be resumed",
      blockTitle: (block: string) => `Experiment block: ${block}`,
      originColab: "Colab",
      originColabTitle: "Trained on Google Colab and imported into this profile",
      epochs: (n: number) => `${n} epoch${n !== 1 ? "s" : ""}`,
      inProgress: "in progress",
    },
    confirm: {
      title: (n: number) => `// permanently delete ${n === 1 ? "1 run" : `${n} runs`}`,
      body: (n: number) =>
        `${n === 1 ? "The run's folder" : "The runs' folders"}, checkpoints and all plots/reports will be removed from disk. This can't be undone.`,
      failed: (failed: number, total: number) =>
        `${failed} of ${total} ${total === 1 ? "run" : "runs"} couldn't be deleted:`,
      unknownError: "unknown error",
      deleting: "Deleting…",
      submit: (n: number) => `🗑 Delete${n > 1 ? ` ${n}` : ""}`,
    },
  },
  leaderboard: {
    viewList: "☰ List",
    viewRanking: "⇅ Ranking",
    viewRankingTitle: "Order each dataset's runs by a metric",
    intro:
      "Each board gathers the runs of one task on the same dataset and orders them by a metric. It is the order of the numbers, not a verdict: with a single seed, the gap between two runs may be seed variance. Tick runs of a board to compare them.",
    runs: (n: number) => `${n} ${n === 1 ? "run" : "runs"}`,
    metricLabel: "metric",
    legend: (metric: string, direction: string) =>
      `Ordered by ${metric} (${direction}). Replicates enter by their mean; a lone run, by its value.`,
    unnamedDataset: "unnamed dataset",
    basis: {
      fingerprint: "same fingerprint",
      path: "by path only",
    },
    basisTitle: {
      fingerprint: (method: string, digest: string) =>
        `Fingerprint ${method} ${digest}: the runs saw the same files, within the limits of that method.`,
      path: "No fingerprint (a run from before the feature, or an unreachable folder): the path is the same, but nothing proves the files did not change between one run and another.",
    },
    unknownTitle: "Unknown dataset",
    unknownBody:
      "Runs with no fingerprint and no dataset path: there is no way to know whether they saw the same data, so they are not ordered.",
    rank: (n: number) => `#${n}`,
    topTag: (mean: boolean) => `1st by ${mean ? "mean" : "value"}`,
    valueNote: {
      mean: (seeds: number) => `mean of ${seeds} seeds`,
      value: "value · 1 seed",
    },
    ciTitle: (low: string, high: string) => `95% CI: [${low}, ${high}]`,
    tickTitle: "Tick to compare",
    openTitle: "Open this run",
    unrankedTitle: "Not ranked",
    reason: {
      "not-finished": "not finished",
      stopped: "cut by a stop",
      comparison: "replicated comparison: no single value (open the group)",
      "no-metric": "did not measure this metric",
    },
    caution: {
      "single-seed":
        "The top two are single-seed runs: the gap between them may be seed variance. To claim a difference, run replicates with several seeds.",
      "single-seed-mixed":
        "One of the top two is a single-seed run with no interval: the gap between them may be seed variance. To claim a difference, run replicates with several seeds.",
      "no-interval":
        "One of the top two carries no confidence interval: it cannot be said whether the gap between them exceeds the variation between seeds.",
      overlap:
        "The 95% intervals of the top two overlap: on this data, their order could flip.",
    },
  },
  runGroup: {
    kind: {
      replicates: "Replicates",
      replicated_comparison: "Replicated comparison",
      kfold: "K-fold",
      sweep: "Sweep",
    },
    card: {
      seeds: (finished: number, requested: number) => `${finished}/${requested} seeds`,
      folds: (finished: number, requested: number) => `${finished}/${requested} folds`,
      trials: (finished: number, requested: number) => `${finished}/${requested} trials`,
      trainings: (finished: number, requested: number) =>
        `${finished}/${requested} trainings`,
      variants: (n: number) => `${n} ${n === 1 ? "variant" : "variants"}`,
      best: (label: string) => `best mean: ${label}`,
      noBest: "no best mean",
      bestTrial: (label: string) => `best trial: ${label}`,
      noBestTrial: "no finished trial",
      stopped: "stopped",
      stoppedTitle: "Stopped on request: the aggregate uses only what finished.",
      showRuns: (n: number) => `▸ ${n} ${n === 1 ? "run in the group" : "runs in the group"}`,
      hideRuns: (n: number) => `▾ ${n} ${n === 1 ? "run in the group" : "runs in the group"}`,
      meanTitle: (n: number, low: string | null, high: string | null) =>
        low !== null && high !== null
          ? `mean of ${n} seeds · 95% CI: [${low}, ${high}]`
          : `mean of ${n} ${n === 1 ? "seed" : "seeds"} · no CI (a single seed)`,
      foldMeanTitle: (n: number, std: string | null) =>
        std !== null
          ? `mean of ${n} folds (validation) · std ${std}`
          : `mean of ${n} ${n === 1 ? "fold" : "folds"} (validation)`,
      deleteTitle: "Delete this group (its seeds stay in the history)",
    },
    deleteNote:
      "Deleting a group removes only its summary: the seeds' runs stay in the history, as ordinary runs.",
    member: {
      title: "Part of a group",
      body: (id: string) => `This seed belongs to the group ${id}.`,
      open: "open the group",
    },
    detail: {
      eyebrow: {
        replicates: "// replicate set",
        replicated_comparison: "// replicated comparison",
        kfold: "// cross-validation (K-fold)",
        sweep: "// hyperparameter sweep",
      },
      metric: "metric",
      seedsAsked: "seeds asked",
      foldsAsked: "folds",
      trainingsDone: "trainings finished",
      alpha: "alpha",
      direction: {
        higher: "higher is better",
        lower: "lower is better",
      },
      report: "report",
      stopped:
        "Stopped on request: the aggregate uses only the trainings that finished; whatever was cut is left out.",
      aggregateTitle: "Aggregate per metric",
      cols: {
        metric: "metric",
        n: "n",
        meanCi: "mean ± 95% CI",
        interval: "95% CI",
        std: "std",
        range: "min – max",
        seed: "seed",
        fold: "fold",
        trial: "trial",
        params: "hyperparameters",
        status: "status",
        run: "run",
        variant: "variant",
        overrides: "overrides",
        seeds: "seeds",
        pair: "pair",
        pairs: "pairs",
        diff: "Δ mean (a−b)",
        test: "test",
        p: "p (raw)",
        holm: "significant after Holm",
        effect: "effect (d)",
      },
      noInterval: "—",
      fewSeeds: (n: number) =>
        `With ${n} ${n === 1 ? "seed" : "seeds"} the 95% CI is wide and fragile: below 5 seeds it does not measure precision.`,
      seedsTitle: "Seeds",
      foldsTitle: "Folds",
      trialsTitle: "Trials",
      mode: "search",
      modes: { grid: "grid", random: "random", optuna: "Optuna", models: "model comparison" } as Record<string, string>,
      best: "best",
      open: "open",
      removed: "removed from the history",
      noRun: "no run",
      status: {
        ok: "finished",
        stopped: "stopped · left out of the aggregate",
        failed: "failed",
      },
      variantsTitle: "Variants",
      bestByMean: "best mean",
      bestNote:
        "It describes the means on the shared seeds; whether the gap exceeds seed variation is what the paired tests below say.",
      noBest: {
        stopped: "the stop left variants unrun",
        "too-few-seeds": "fewer than 2 seeds finished in common by the variants",
      },
      rankingSeeds: (seeds: string) => `ranked on seeds ${seeds}`,
      notRun: (labels: string) => `Not run (stopped): ${labels}`,
      skipped: (labels: string) => `Too few seeds to test: ${labels}`,
      testsTitle: "Paired tests",
      noTests: "No pair of variants could be tested.",
      yes: "yes",
      no: "no",
      tests: {
        paired_t: "paired t",
        wilcoxon: "Wilcoxon",
      },
      effects: {
        negligible: "negligible",
        small: "small",
        medium: "medium",
        large: "large",
      },
      underpowered: "few seeds",
      underpoweredTitle: (pairs: number, floor: string) =>
        `With ${pairs} pairs the smallest possible p is ${floor}, above alpha: no result, however consistent, could be significant.`,
      testsNote:
        "p is the raw test value. The \"significant after Holm\" column is already corrected for the set of comparisons; it is the one that counts.",
    },
  },
  taskPanel: {
    sweep: "Sweep",
    replicates: "Replicates",
    pretrained: "Pretrained weights",
    backbone: "Backbone",
    imageNet: "ImageNet",
    // The hint under "pretrained weights" in the segmentation and regression forms.
    transferHint: "pretrained backbone (torchvision)",
    training: {
      title: "Training",
      epochs: "Epochs",
      batchSize: "Batch size",
      batchSizeHint: "any integer",
      learningRate: "Learning rate",
      loss: "Loss",
      seed: "Seed",
      optimizer: "Optimizer",
      earlyStop: "Early stop",
      earlyStopHint: "patience",
      deterministic: "Deterministic",
      deterministicHint: "reproducible",
    },
    transfer: {
      title: "Transfer learning",
      mode: "Mode",
      full: "Full",
      featureExtraction: "Feature extr.",
      fineTuning: "Fine-tuning",
      backboneLr: "Backbone LR ×",
      backboneLrHint: "backbone LR = base LR × this",
    },
    dataset: {
      baseDir: "Base folder",
      baseDirHint: "dataset root",
      browse: "📁 Browse",
      imageSize: "Image size",
      trainSplit: "Train subfolder",
      valSplit: "Validation subfolder",
      testSplit: "Test subfolder",
      optional: "optional",
      imagesSubdir: "Image subfolder",
    },
  },
  segmentationPanel: {
    namePlaceholder: "segmentation_001",
    pixelAcc: "Pixel accuracy",
    ignoreIndexCollision: (index: number, maxClass: number) =>
      `ignore_index (${index}) collides with a real class index (0…${maxClass}). Use a value outside that range (e.g. 255 or -1).`,
    model: {
      title: "Model · segmentation",
      architecture: "Architecture",
      architectureHint: "dense head",
      numClasses: "Number of classes",
      numClassesHint: "includes background",
      pretrainedHint: "ImageNet backbone",
    },
    lossHint: "per-pixel (CE) or overlap (Dice)",
    dataset: {
      title: "Dataset (images + masks)",
      baseDirPlaceholder: "…/dataset (train/{images,masks}, val/…)",
      imagesSubdirHint: "per split",
      masksSubdir: "Mask subfolder",
      masksSubdirHint: "PNG · class index per pixel",
      ignoreIndex: "ignore_index",
      ignoreIndexHint: "void pixels",
    },
  },
  regressionPanel: {
    namePlaceholder: "regression_001",
    model: {
      title: "Model · regression",
      backboneHint: "CNN or transformer",
      numTargets: "Number of targets",
      numTargetsHint: "derived from the target columns",
    },
    lossHint: "criterion",
    dataset: {
      title: "Dataset (CSV manifest)",
      baseDirPlaceholder: "…/dataset (train.csv, val.csv, images/)",
      imagesDirHint: "relative to the base folder",
      imageColumn: "Image column",
      imageColumnHint: "column name in the CSV",
      targetColumns: "Target columns",
      targetColumnsPlaceholder: "target  or  x,y,z",
      targetColumnsHint: "comma-separated",
      trainCsv: "Train CSV",
      valCsv: "Validation CSV",
      testCsv: "Test CSV",
    },
  },
  anomalyPanel: {
    namePlaceholder: "anomaly_001",
    imageF1: "F1 (image-level)",
    compareLabel: "Compare architectures",
    compareNote:
      "Trains each architecture with the same config and measures all of them on the test set. It is a comparison report, not a selection: anomaly has no labelled validation split, and picking the model by the test set would make its number optimistic. To claim one is better, run replicates of each.",
    model: {
      title: "Model · anomaly",
      method: "Method",
      methodHint: "reconstruction or memory bank",
      backboneHint: "frozen feature extractor",
      coresetRatio: "Coreset ratio",
      coresetRatioHint: "memory bank subsample",
      pretrainedBackbone: "Pretrained backbone",
      latentDim: "Latent dim",
      latentDimHint: "autoencoder bottleneck",
    },
    epochsHint: "ignored by PatchCore",
    threshold: "Threshold percentile",
    thresholdHint: "percentile of normal-image scores",
    dataset: {
      title: "Dataset (MVTec layout · trains on normal images only)",
      baseDirPlaceholder: "…/category (train/good, test/good, test/<defect>)",
      normalDir: "Normal subfolder",
      normalDirHint: "label 0 (e.g. good)",
    },
  },
  datasetDownload: {
    title: "Download a dataset (online → local folder)",
    cancel: "cancel",
    open: "+ download dataset",
    provider: "Provider",
    torchvision: {
      dataset: "Dataset",
      datasetHint: "built-in",
      limit: "Max images per class",
      limitPlaceholder: "blank = all",
      limitHint: "optional",
    },
    roboflow: {
      dataset: "Dataset or URL",
      datasetPlaceholder: "workspace/project — or paste the URL",
      version: "Version",
      apiKey: "API key",
      apiKeyHint: "app.roboflow.com → Settings → API Keys",
      format: "Format",
      formatPlaceholder: "folder",
      formatHint: "folder for classification",
    },
    kaggle: {
      dataset: "Dataset (owner/slug)",
      datasetPlaceholder: "zalando-research/fashionmnist",
      token: "API token",
      tokenPlaceholder: "KGAT_…",
      tokenHint: "kaggle.com → Settings → API → Create New Token",
    },
    huggingface: {
      dataset: "Dataset (HF Hub ID)",
      datasetPlaceholder: "owner/dataset",
      token: "Access token",
      tokenHint: "optional — only for private datasets",
    },
    outDir: "Output folder",
    outDirPlaceholder: "…/datasets/downloaded",
    outDirHint: "where to save",
    browse: "📁 Browse",
    download: "▶ Download",
    downloading: "Downloading…",
    needDatasetAndFolder: "Enter the dataset and the output folder.",
    downloadingWait: "Downloading… (this may take a while)",
    done: (images: number, dir: string) => `${images} ${images === 1 ? "image" : "images"} in ${dir}`,
    failed: "Failed to download the dataset.",
    classes: (n: number) => ` · ${n} ${n === 1 ? "class" : "classes"}`,
  },
  credentialField: {
    typeFirst: "Enter a key before saving.",
    saved: "Saved — no need to type it again.",
    saveFailed: "Failed to save.",
    removed: "Key removed from this computer.",
    removeFailed: "Failed to remove.",
    savedPlaceholder: (masked: string) => `saved: ${masked}`,
    savedHint: "leave blank to use the saved one",
    replace: "↻ Replace",
    save: "💾 Save",
    forget: "Forget",
  },
  datasetPicker: {
    analyzing: "Scanning subfolders…",
    detectFailed: "Failed to detect the dataset splits.",
    opening: "Opening the system folder picker…",
    cancelled: "Selection canceled.",
    picked: (path: string) => `Selected folder: ${path}`,
    pickFailed: "Failed to open the folder picker.",
    baseDir: "Dataset base folder",
    baseDirPlaceholder: "e.g. C:/datasets/coffee  or  /home/user/data",
    baseDirHint: "Root folder that holds the train, validation and test sets.",
    browseTitle: "Open the system's native folder picker (returns the absolute path)",
    browse: "📁 Browse",
    trainSubdir: "Train subfolder",
    valSubdir: "Validation subfolder",
    testSubdir: "Test subfolder",
    manual: (value: string) => `${value} (manual)`,
  },
  sampleDataset: {
    button: "use the sample dataset",
    title: "Creates datasets/exemplo-<task>/ (or reuses it) and fills the base folder",
    failed: (reason: string) => `Could not create the sample dataset: ${reason}`,
  },

  datasetStats: {
    splits: {
      train: "train",
      val: "validation",
      test: "test",
    },
    missing: "missing",
    imbalanced: "⚠ imbalanced",
    samples: (split: string) => `// samples (split: ${split}) — label sanity check`,
    analyzing: "Analyzing dataset…",
    noClasses: "No classes found.",
    classMap: "class mapping (ImageFolder):",
    appliedBinaryTitle: "Detected and applied to the config: task=binary, num_classes=1",
    appliedMulticlassTitle: (classes: number) =>
      `Detected and applied to the config: task=multiclass, num_classes=${classes}`,
    appliedBinary: "binary applied",
    appliedMulticlass: (classes: number) => `multiclass · ${classes} applied`,
    distribution: "// dataset distribution",
    sampleAlt: (className: string) => `${className} sample`,
    noImages: "no images",
  },
  detectionDatasetStats: {
    noSplits: "No YOLO splits found (images/<split>).",
    classMap: "class mapping (YOLO):",
    appliedTitle: (classes: number) => `Detected and applied to the config: num_classes=${classes}`,
    applied: (classes: number) => `${classes} ${classes === 1 ? "class" : "classes"} applied`,
    exampleAlt: (className: string, n: number) => `${className} — example ${n}`,
    distribution: "// annotation distribution (instances)",
    layoutTitle: "YOLO layout detected in this split",
    images: (n: number) => `${n} img`,
    boxes: (n: number) => `${n} ${n === 1 ? "box" : "boxes"}`,
    unlabeled: (n: number) => `${n} unlabeled`,
  },
  taskDatasetStats: {
    interpolated: (ids: number) =>
      `⚠ ${ids} distinct values in the sampled masks — they look interpolated (anti-aliased). Use masks with one class index per pixel (NEAREST resampling).`,
    maskIds: "mask values (sample):",
    voidTitle: "likely ignore_index (void)",
    applyClasses: (n: number) => `🎯 apply ${n} ${n === 1 ? "class" : "classes"}`,
    pairing: "// image ↔ mask pairing",
    pairs: (n: number) => `${n} ${n === 1 ? "pair" : "pairs"}`,
    pairCounts: (images: number, masks: number) =>
      `${images} img · ${masks} ${masks === 1 ? "mask" : "masks"}`,
    unpaired: (images: number, masks: number) =>
      `⚠ ${images} img without mask · ${masks} ${masks === 1 ? "mask" : "masks"} without img`,
    anomalyKicker: "// normal vs. anomalous distribution",
    trainNormal: "train (normal)",
    testNormal: "test · normal",
    testAnomalous: "test · anomalous",
    images: (n: number) => `${n} img`,
    missingTestDir: "test folder missing",
    regressionKicker: "// manifest & target distribution",
    rows: (n: number) => `${n} ${n === 1 ? "row" : "rows"}`,
    missingColumns: (columns: string) => `⚠ missing columns: ${columns}`,
    missingImages: (missing: number, checked: number) =>
      `⚠ ${missing}/${checked} ${missing === 1 ? "image" : "images"} not found`,
    target: (column: string, mean: string, min: string, max: string, n: number) =>
      `${column}: μ ${mean} · [${min}, ${max}] · n=${n}`,
  },
  preprocessing: {
    needBaseDir: "Set the dataset base folder before generating a preview.",
    kicker: "// preprocessing (filters)",
    active: (n: number) => `${n} ${n === 1 ? "filter" : "filters"} active in training`,
    activeTitle:
      "These filters are applied during training, before augmentation and normalization",
    clearTitle: "Remove all filters from the pipeline",
    clear: "clear",
    addFilter: "+ add filter",
    generating: "Generating…",
    preview: "▶ Preview",
    empty: 'Pipeline is empty — click "+ add filter".',
    emptyNote:
      "The pipeline configured here runs **before** augmentation and normalization, on every split (train / val / test).",
    original: "Original",
    final: "Final",
    moveUp: "Move up",
    moveDown: "Move down",
    remove: "Remove",
  },
  augmentPreview: {
    needBaseDir: "Set the dataset base folder first.",
    generating: "Generating…",
    button: "🎲 augmentation preview",
    active: "active:",
    noneActive: "no augmentation active — resize only",
    original: "original",
    variant: (n: number) => `variant ${n}`,
  },
  transforms: {
    image: "Image",
    normalizeHint: "R, G, B — applied to train, validation and test",
    augmentation: "Data augmentation",
    augmentToggle: "Augmentation",
    augmentHint: "training only; when off, the values below are saved for later",
    flipHint: "training only",
    rotationHint: "0 = off",
  },
  toggle: {
    on: "ON",
    off: "OFF",
  },
  tasks: {
    classification: {
      label: "Classification",
      description: "Sort images into discrete classes",
    },
    detection: {
      label: "Object detection",
      description: "Locate and classify objects with bounding boxes",
    },
    regression: {
      label: "Regression",
      description: "Estimate continuous values from the inputs",
    },
    segmentation: {
      label: "Segmentation",
      description: "Pixel-level masks for each category",
    },
    anomaly: {
      label: "Anomaly",
      description: "Detect defects by training on normal images only",
    },
  },
  regressionModels: {
    resnet18: "light · 11.7M",
    resnet34: "21.8M",
    resnet50: "imagenet · 25.6M",
    resnet101: "deep · 44.5M",
    efficientnet_b1: "efficient · 7.8M",
    efficientnet_b7: "large · 66M",
    vgg16: "classic · 138M",
    vgg19: "144M",
    alexnet: "baseline · 61M",
    vit_b_16: "transformer · 86M",
    swin_t: "transformer · 28M",
    convnext_tiny: "modern · 28M",
  },
  segmentationModels: {
    deeplabv3_resnet50: "imagenet · 42M",
    deeplabv3_resnet101: "deep · 61M",
    deeplabv3_mobilenet_v3_large: "light · 11M",
    fcn_resnet50: "32.9M",
    fcn_resnet101: "51.9M",
    lraspp_mobilenet_v3_large: "mobile · 3.2M",
    unet: "classic · 31M",
  },
  anomalyModels: {
    autoencoder: "reconstruction · trainable",
    patchcore: "memory bank · backbone",
    backbones: {
      resnet18: "light",
      wide_resnet50_2: "patchcore default",
    },
  },
  queueFormat: {
    strategies: {
      simple: "single run",
      kfold: "K-fold",
      transferLearning: "transfer learning",
      gridSearch: "grid search",
      randomSearch: "random search",
      sweep: "sweep",
      replicates: "replicates",
      comparison: "comparison",
      replicatedComparison: "replicated comparison",
    },
  },
  gridAxis: {
    notAnOption: "not one of the options",
    invalidValue: "invalid value",
    mustBeInteger: "must be an integer",
    mustBeAbove: (min: number) => `must be > ${min}`,
    mustBeAtLeast: (min: number) => `must be ≥ ${min}`,
    mustBePowerOfTwo: "must be a power of 2",
  },
  runNotify: {
    completedTitle: (label: string) => `Training finished — ${label}`,
    completedBody: "Open VisionForge to see the results.",
    failedTitle: (label: string) => `Training failed — ${label}`,
    failedBody: "Open VisionForge to see the error.",
  },
  yamlConfig: {
    cannotRead: (reason: string) => `Failed to read the YAML file: ${reason}`,
    invalidFile: (reason: string) => `Invalid YAML file: ${reason}`,
    notMapping:
      "the content must be a mapping (key: value), not a single value or a list.",
    expectedObject: "Expected an object.",
    expectedArray: "Expected a list.",
    requiredMissing: "Required field is missing.",
    mustBeOneOf: (options: string) => `Must be one of: ${options}.`,
    expectedBoolean: "Expected a boolean.",
    expectedInteger: "Expected an integer.",
    expectedNumber: "Expected a number.",
    expectedString: "Expected a string.",
  },
  datasetIdentity: {
    noFingerprint: "one of the runs has no fingerprint (it predates 2026-07-26)",
    differentMethod: "the two runs were fingerprinted with different methods",
  },
  customTasks: {
    fallbackDescription: "Researcher-defined task",
  },
  replicatesForm: {
    needTwoSeeds: "enter at least 2 seeds",
    duplicateSeeds: "duplicate seeds",
  },
  experiment: {
    resultFetchFailed: "Failed to fetch the experiment results.",
    failedNoDetail: "The experiment failed without an error message.",
    connectionLost: "Lost connection to the server while polling.",
    validationFailed: (n: number) =>
      `${n} ${n === 1 ? "field has a validation error" : "fields have validation errors"}. Check the highlighted ${n === 1 ? "field" : "fields"} in the form.`,
    alreadyRunning: "An experiment is already running. Wait for it to finish.",
    unexpected: (message: string) => `Unexpected error: ${message}`,
    unknown: "Unknown error while starting the experiment.",
    sections: {
      preprocessing: "Preprocessing",
      steps: "Filter",
      scheduler: "Scheduler",
      device: "Device",
    },
  },
  // The six-step configuration (components/StepShell.tsx, ADR-156).
  stepShell: {
    steps: {
      data: "Data",
      strategy: "Strategy",
      model: "Model",
      training: "Training",
      image: "Image & augmentation",
      review: "Review",
    },
    leads: {
      data: "Where the images come from and what was found in them.",
      strategy: "How many runs and how to compare them.",
      model: "What trains and where it starts from.",
      training: "How long and how to train; the rest is under Advanced.",
      image: "How each image reaches the model, with a preview.",
      review: "Check it, name it and train from the bar below.",
    },
    status: {
      default: "default",
      changes: (n: number) => `${n} ${n === 1 ? "change" : "changes"}`,
      missing: "missing",
      errors: (n: number) => `${n} ${n === 1 ? "error" : "errors"}`,
    },
    stepLabel: (i: number, n: number) => `step ${i} of ${n}`,
    navLabel: "Configuration steps",
    prev: "← Back",
    next: "Next →",
    goReview: "Review →",
    groups: {
      mechanics: "// mechanics",
      schedule: "LR schedule",
      loss: "Loss weights",
      color: "Color",
      geometry: "Geometry",
      composition: "Composition",
      other: "Other",
    },
    transferLocked:
      "Transfer learning is a single run: the backend runs one block at a time. Set the strategy back to Single run to turn it on.",
    review: {
      nameTitle: "Experiment name",
      nameHint: "suggested from dataset, model and epochs; edit freely",
      useSuggestion: "use suggestion",
      trialNote: (keys: string) => `each trial gets the value of ${keys} in its name`,
      edit: "edit",
      ready: "All set. The Train button in the bar below is enabled.",
      blocked: "Choose the dataset in the Data step to enable Train.",
      errorsTitle: "Problems in the last attempt",
      noChanges: "default values",
      lrPlan: {
        title: "Learning rate of each trial",
        model: "model",
        optimizer: "optimizer",
        multiplier: "multiplier",
        effectiveRate: "effective LR",
        perModel: "per model, see below",
        unavailable: "Failed to work out the learning rate of each trial.",
        combination: (architecture: string, optimizer: string, rate: number) =>
          `${architecture} with ${optimizer} at ${rate}: flagged combination.`,
        measured: (facts: LrPlanEvidence) =>
          `Measured: ${facts.measuredOn} with Adam at 1e-3 ${
            facts.outcome === "collapse" ? "predicted a single class" : "learned little"
          } (accuracy ${facts.accuracy.toFixed(2)} on 4 classes).`,
        siblingNotMeasured: (architecture: string) => `${architecture} was not measured.`,
        notMeasured: "This family was not measured.",
      },
    },
  },
  experimentHeader: {
    title: "Experiment",
    nameLabel: "Experiment name",
    nameHint: "names the output folder and the history entry",
    exportTitle: "Export the current configuration as a .yaml file",
    exportYaml: "↓ Export YAML",
    importTitle: "Import a configuration from a .yaml file",
    importYaml: "↑ Import YAML",
    imported: (file: string) => `✓ ${file} imported`,
    strategyTitle: "Experiment strategy",
    mode: "Mode",
    hints: {
      simple: "one run with the config below (Train button)",
      cv: "K folds over the training set → per-fold metrics + mean ± std",
      sweep: "grid / random / optuna over the config below",
      replicates: "same config, N seeds → mean ± 95% CI",
      compare: "one run per architecture, same config → comparison table",
    },
  },
  cvCard: {
    title: "Cross-validation (K-fold)",
    description:
      "Splits the training set into K folds: each fold trains a fresh model on the other K−1 folds and evaluates it on the held-out one (never augmented). The test split is not used.",
    folds: "Folds",
    shuffle: "Shuffle",
    shuffleHint: "before splitting",
    foldSeed: "Fold seed",
    summary: (folds: number) => `${folds} folds · run it with the button below`,
  },
  compareModelsCard: {
    title: "Architecture comparison · report",
    metric: "Order by",
    summary: (n: number) =>
      `${n} architecture${n === 1 ? "" : "s"} · run it with the button below`,
    blocked: "Select at least 2 architectures to compare.",
  },
  replicatesCard: {
    title: "Multi-seed replicates · statistical rigor",
    description:
      "Trains the same config N times with different seeds and aggregates each metric as mean ± 95% CI (Student's t). A single run is one sample from a distribution — replicates make the number defensible.",
    seeds: "Seeds",
    automatic: "Automatic",
    explicit: "Explicit",
    automaticHint: "consecutive, starting from training.seed",
    explicitHint: "exact list, reproducible",
    count: "Replicates",
    seedList: "Seeds (comma-separated)",
    seedCount: (n: number) => `${n} seeds`,
    headlineMetric: "Headline metric",
    summary: (seeds: number) =>
      `${seeds} seed${seeds === 1 ? "" : "s"} · run it with the button below`,
    blocked: "Needs at least 2 valid seeds to run.",
  },
  sweepCard: {
    title: "Hyperparameter sweep · advanced mode",
    description: (paths: string) =>
      `Sweeps hyperparameters by dot-path (e.g. ${paths}) and ranks the trials by the chosen metric. Grid = Cartesian product; Random = random sampling; Optuna = adaptive TPE search.`,
    strategy: "Strategy",
    presetTitle: "preset · architectures → model.name axis",
    compareArchitectures: (n: number) => `⇒ compare ${n} architecture${n === 1 ? "" : "s"}`,
    parameter: "Parameter (dot-path)",
    values: "Values (comma-separated)",
    distribution: "Distribution",
    kinds: {
      uniform: "Uniform",
      logUniform: "Log-uniform",
      choice: "Choice",
    },
    options: "Options (comma-separated)",
    low: "low",
    high: "high",
    removeParameter: "Remove parameter",
    addParameter: "+ add parameter",
    rankingMetric: "Ranking metric",
    trials: "Trials",
    seed: "Seed",
    summary: (trials: number) =>
      `${trials} trial${trials === 1 ? "" : "s"} · run it with the button below`,
    blocked: "Add a parameter with values to sweep.",
  },
  schemaForm: {
    sections: {
      training: "Training",
      data: "Dataset",
      transforms: "Augmentation & normalization",
      preprocessing: "Preprocessing (filters)",
      scheduler: "Learning rate scheduler",
      output: "Output",
      model: "Model",
    },
    taskParameters: "Task parameters",
    listHint: "comma-separated values",
  },
  customTaskPanel: {
    schemaLoadFailed: "Failed to load the schema for",
    schemaLoadHint:
      "Check the file in `user_tasks/` — an import error is logged by the server and the task is left without a form.",
    loadingForm: (task: string) => `Loading the form for ${task}…`,
  },
  customTaskManage: {
    heading: "// manage this task",
    close: "close",
    options: "⚙ options",
    hideTitle: "Hides the tab; the file stays in user_tasks/",
    hide: "👁 Hide tab",
    hideNote: "reversible — the file stays",
    hideFailed: "Failed to hide the task.",
    delete: "🗑 Delete from disk",
    deleteNote: "deletes the .py file you wrote — can't be undone",
    confirmBefore: "This deletes the .py file that defines",
    confirmAfter: "from disk; this can't be undone. To confirm, type the task key:",
    typeKeyTitle: "Type the task key exactly to enable this button",
    confirmLabel: "Task key to confirm",
    deleting: "Deleting…",
    deleteForever: "Delete permanently",
    deleteFailed: "Failed to delete the task.",
  },
  queueOverlay: {
    readFailed: "Failed to read the queue.",
    cancelFailed: "Failed to cancel that run.",
    kicker: "// queue",
    title: "Run queue",
    description:
      "One run at a time in each queue: this machine's and each Colab account's, which run side by side. Submit as many as you like — they run on their own, in the order they were submitted.",
    emptyTitle: "The GPU is free and nothing is waiting.",
    emptyHint: "Runs you submit while another is training wait here.",
    position: (n: number) => ordinal(n),
    laneLocal: "Local · this machine",
    laneColab: (account: string) => `Colab · ${account}`,
    laneEmpty: "nothing here",
    originLocal: "local",
    originColab: "Colab",
    running: "running",
    waiting: (waited: string) => `waiting ${waited}`,
    watch: "🔬 view training",
    removeTitle: "Remove from the queue (does not affect runs already started)",
    stopTitle: "Stop this run (the work done so far is kept)",
    stopUnavailable:
      "This run does not accept a stop request (a custom task that owns its training loop, for example): it goes on to the end.",
    stopRefused:
      "The server refused the request: this run does not accept a stop and goes on to the end.",
    stop: "■ stop",
    remove: "🗑 remove",
  },
  trainingOverlay: {
    blocks: {
      modelComparison: "Model comparison",
      crossValidation: "K-fold CV",
      replicates: "Multi-seed replicates",
    },
    phases: {
      extractingFeatures: "extracting features",
      buildingBank: "building the memory bank",
      scoring: "scoring",
    },
    initializing: (runId: string) => `> initializing runtime · ${runId}`,
    loadingDataset: "> loading dataset…",
    trainingFailed: "training failed",
    trainingComplete: "training complete",
    queued: (task: string) => `queued · ${task}`,
    training: (task: string) => `training · ${task}`,
    starting: "starting…",
    queuedNote: (position: number | undefined, queued: number | undefined) =>
      `${position ? `waiting for the GPU — ${ordinal(position)} in line` : "waiting for the GPU"}${
        typeof queued === "number" && queued > 1 ? ` · ${queued} runs waiting` : ""
      }. The run starts automatically when its turn comes.`,
    queuedColabNote: (account: string, position: number | undefined) =>
      `${position ? `waiting for ${account}'s Colab — ${ordinal(position)} in line` : `waiting for ${account}'s Colab`}. The run starts automatically when that account's previous run ends.`,
    driveConsentTitle: "Colab asks for access to your Google Drive",
    driveConsentBody: (attempt: number, attempts: number) =>
      `Allow it in a new tab and come back to this window; the run goes on by itself. Attempt ${attempt} of ${attempts}.`,
    driveConsentOpen: "Allow Drive access",
    driveConsentDone: "I already allowed",
    driveConsentSending: "Checking the permission…",
    driveConsentError: "The server did not get the answer. Try \"I already allowed\" again.",
    // The status poll is not reaching the server; the run goes on there (ADR-133).
    reconnecting: "reconnecting…",
    reconnectingNote: "The server did not answer. Trying again — the training goes on on the server.",
    connectionLost: "no connection to the server",
    connectionLostNote:
      "No answer for over 30 s. The training may still be going on the server; this panel picks it up again by itself once the server answers, and the History shows the result.",
    errorDetail: "Error details",
    unknownError: "Unknown error — check the server logs.",
    logUnknownError: "unknown error",
    queueBanner: (block: string, runs: number | undefined) =>
      `⛓ multi-run block · ${block}${runs && runs > 1 ? ` · ${runs} runs` : ""}`,
    queueBannerBody:
      'This block trains several models one after another. The progress bar tracks the current trial; the combined result appears under "View results" when it finishes.',
    pipeline: (n: number) => `⚗ active pipeline · ${n} filter${n === 1 ? "" : "s"}`,
    minimize: "Minimize",
    viewResults: "↗ View results",
    stop: "■ Stop",
    stopTitle: "Stop the run at the end of the epoch in progress",
    stopTitlePhase:
      "Stop PatchCore (building the memory bank is not interrupted part-way)",
    stopUnavailable:
      "This run does not accept a stop request (a custom task that owns its training loop, for example): it goes on to the end.",
    stopConfirm: {
      epoch:
        "Stop this run? It finishes the epoch in progress and keeps the best checkpoint and the history so far. If it stops before the last epoch, you can resume it from the History.",
      trial:
        "Stop this search? The trial in progress finishes its current epoch and no other starts. That cut trial stays out of the ranking, and the finished ones are kept. A stopped search cannot be resumed.",
      fold:
        "Stop the cross-validation? The fold in progress finishes its current epoch and no other starts. The mean and standard deviation use only the finished folds; the cut fold is left out. A stopped cross-validation cannot be resumed.",
      model:
        "Stop the comparison? The model in progress finishes its current epoch and no other starts. The ranking uses only the finished models; the cut model is left out. A stopped comparison cannot be resumed.",
      replicate:
        "Stop the replicates? The replicate in progress finishes its current epoch and no other starts. The mean and interval use only the finished replicates; the cut replicate is left out. A stopped set cannot be resumed.",
      phase:
        "Stop PatchCore? During feature extraction nothing is kept. If it is already building the memory bank, that finishes and is saved, and scoring is skipped — you can score it later with “+ test” in the History.",
      vm: "Stop the training on Colab? The VM is released right away: the epoch in progress is lost, and nothing trained there comes back to this machine yet.",
    },
    stopConfirmCustom: {
      epoch:
        "Stop this task? It finishes the epoch in progress and keeps what has been trained so far. A custom task cannot be resumed from the History.",
      trial:
        "Stop this search? No new trial starts. The trial in progress finishes its current epoch — or runs to the end, if the task owns its training loop. A cut trial stays out of the ranking, and the finished ones are kept. A stopped search cannot be resumed.",
      replicate:
        "Stop the replicates? No new replicate starts. The replicate in progress finishes its current epoch — or runs to the end, if the task owns its training loop. A cut replicate stays out of the mean and the interval, which use only the finished ones. A stopped set cannot be resumed.",
    },
    stopConfirmYes: "Stop the run",
    stopConfirmNo: "Keep training",
    stopSending: "Stopping…",
    stopRequested:
      "Stop requested: the run finishes the epoch in progress and nothing new starts.",
    stopRequestedPhase:
      "Stop requested: PatchCore stops between phases (a memory bank being built finishes first).",
    stopFailed: "Failed to stop the run.",
    stopAlreadyEnded: "The run had already ended when the request arrived.",
    stopRefused:
      "The server refused the request: this run does not accept a stop and goes on to the end.",
    stopUnconfirmed:
      "The Stop button unlocks once it is known where this run can stop.",
    stopped: "Training stopped",
    stoppedLog: (epoch: number | null, total: number | null) =>
      epoch === 0
        ? "stopped before the first epoch"
        : `stopped${epoch !== null && total !== null ? ` at epoch ${epoch}/${total}` : ""} · best checkpoint kept`,
    stoppedUnitsLog: (
      unit: "trial" | "fold" | "model" | "replicate",
      finished: number,
      planned: number | null,
      stopped: number | null,
    ) => {
      const label = {
        trial: "trials finished",
        fold: "folds finished",
        model: "models finished",
        replicate: "replicates finished",
      }[unit];
      return `stopped · ${label}: ${finished}${planned !== null ? `/${planned}` : ""}${stopped !== null && stopped > 0 ? ` · left out: ${stopped}` : ""}`;
    },
    stoppedPhaseLog: (bankKept: boolean): string =>
      bankKept
        ? "stopped · memory bank saved, not scored (score it later with “+ test”)"
        : "stopped during extraction · nothing was kept",
    stopTooLate: "> the stop cut nothing: the run finished normally",
  },
  liveResults: {
    tab: "Results",
    empty: "The curves appear once the first epoch ends.",
    noEpochsPatchcore:
      "PatchCore does not train in epochs: it builds its memory bank in a single pass. The result comes at the end.",
    best: (epoch: number, metric: string, value: string) =>
      `Best epoch: ${epoch} (${metric} ${value})`,
    bestShort: (epoch: number, metric: string, value: string) =>
      `epoch ${epoch} · ${metric} ${value}`,
    bestNote: "the saved checkpoint is from this epoch",
    lossChart: "Loss",
    gradcamTitle: (epoch: number, best: boolean) =>
      `Validation Grad-CAM · epoch ${epoch}${best ? " · best so far" : ""}`,
    gradcamNote: "where the model looked to decide; the same images every time",
    gradcamCaption: (actual: string, predicted: string) =>
      actual === predicted ? `${actual} ✓` : `${actual} → read as ${predicted}`,
    lrChart: "Learning rate",
    confusionTitle: (epoch: number) => `Validation confusion matrix · epoch ${epoch}`,
    confusionAxes: "rows: true class · columns: predicted class",
    topTitle: (epoch: number) => `Most frequent validation confusions · epoch ${epoch}`,
    topRow: (actual: string, predicted: string) => `${actual} read as ${predicted}`,
    unitsTitle: "Finished trainings",
    changesCol: "What changes",
    epochsCol: "Epochs",
    bestCol: "Best",
  },
  trainingStory: {
    tab: "How it works",
    logTab: "Log",
    tabsLabel: "Training sheet content",
    stripLabel: "Training stages",
    caption: "illustrative animation, slowed down: only the stage and the numbers are real",
    figure: (family: string) => `Illustrative drawing of a network: ${family}`,
    overfitting: (since: number) =>
      `Sign of overfitting: since epoch ${since}, the validation loss rises every epoch while the training loss falls. The model may be memorising the training set instead of learning what holds for new images.`,
    families: {
      cnn: "convolutional network (CNN)",
      transformer: "transformer: patches that attend to one another",
      encoderDecoder: "encoder–decoder, like the U-Net",
      detector: "detector: backbone, neck and heads that give boxes",
      autoencoder: "autoencoder: compresses and rebuilds",
      patchcore: "PatchCore: extractor, memory bank and scoring",
    },
    stageNames: {
      queued: "queued",
      preparing: "data",
      training: "training",
      testing: "test",
      done: "done",
      interrupted: "interrupted",
      failed: "failed",
    },
    stages: {
      queued: "Queued: training starts by itself when the GPU is free.",
      preparing: "Loading the data and building the model; no epoch has run yet.",
      training:
        "The model sees the data, gets it wrong, and the error flows back adjusting the weights; at the end of each epoch, validation measures how much it learned.",
      testing:
        "Training is over: the model is measured on the test split, which it has never seen, and the report is written.",
      done: "Done: the full results are under “View results”.",
      interrupted: "Interrupted on request: whatever was already saved is still available.",
      failed: "Training failed: the error detail is in the box above and in the log.",
    },
    phases: {
      extractingFeatures:
        "A network that is already trained reads the normal images and extracts their features; nothing is adjusted.",
      buildingBank:
        "The most representative features of what is normal are stored in the memory bank.",
      scoring:
        "Each test image is compared with the bank: the farther from what is normal, the higher the anomaly score.",
    },
    numbers: {
      epoch: "epoch",
      epochOf: (epoch: number, total: number | null) =>
        total === null ? `${epoch}` : `${epoch}/${total}`,
      trainLoss: "train loss",
      valLoss: "validation loss",
      phase: "phase",
      metrics: {
        accuracy: "accuracy",
        map50: "mAP50",
        r2: "R²",
        miou: "mIoU",
        auroc: "AUROC",
        primary: "main metric",
      },
    },
    units: {
      trial: (index: number, total: number) => `trial ${index}/${total}`,
      fold: (index: number, total: number) => `fold ${index}/${total}`,
      model: (index: number, total: number) => `model ${index}/${total}`,
      replicate: (index: number, total: number) => `replicate ${index}/${total}`,
    },
    // A finished run can show its stages again, in slow motion (ADR-118).
    replay: { button: "replay animation", playing: "replaying…" },
    legend: {
      forward: "data and forward pass",
      backward: "gradient flowing back",
      validation: "validation",
      patchForward: "features stored",
      patchScore: "scoring",
    },
  },
  compareRuns: {
    back: "← history",
    comparing: (n: number) => `Comparing ${n} runs`,
    loading: "loading runs…",
    loadFailed: "Failed to load runs.",
    verdict: {
      same: "same data",
      different: "different data",
      unknown: "can't be verified",
    },
    metric: "Metric",
    device: "Device",
    field: "Field",
    metrics: {
      best_val_loss: "Best val loss",
      best_epoch: "Best epoch",
      total_epochs: "Epochs trained",
      test_accuracy: "Accuracy (test)",
      test_f1: "F1 (test)",
      test_precision: "Precision (test)",
      test_recall: "Recall (test)",
      test_auc_roc: "AUC-ROC (test)",
      map50_95: "mAP@50-95 (val)",
      map50: "mAP@50 (val)",
      precision: "Precision (val)",
      recall: "Recall (val)",
      test_r2: "R² (test)",
      test_rmse: "RMSE (test)",
      test_mae: "MAE (test)",
      r2: "R² (val)",
      rmse: "RMSE (val)",
      mae: "MAE (val)",
      test_miou: "mIoU (test)",
      test_dice: "Dice (test)",
      test_pixel_acc: "Pixel accuracy (test)",
      val_miou: "mIoU (val)",
      val_dice: "Dice (val)",
      val_pixel_acc: "Pixel accuracy (val)",
      miou: "mIoU (val)",
      dice: "Dice (val)",
      pixel_acc: "Pixel accuracy (val)",
      test_auroc: "AUROC (test)",
      test_image_f1: "Image F1 (test)",
      test_threshold: "Decision threshold (test)",
      auroc: "AUROC (val)",
      image_f1: "Image F1 (val)",
      threshold: "Decision threshold (val)",
    },
    direction: {
      higher: "higher is better",
      lower: "lower is better",
    },
    extreme: {
      highest: "highest value",
      lowest: "lowest value",
    },
    seedNote: {
      "single-seed":
        "Each run has a single seed: the gap between them can be seed variation. To claim a difference, run replicates with several seeds.",
      "single-seed-mixed":
        "The runs that are not replicate means have a single seed: a gap involving them can be seed variation. To claim a difference, run replicates with several seeds.",
    },
    group: {
      tag: (n: number) => `mean of ${n} ${n === 1 ? "seed" : "seeds"}`,
      cellTitle: (n: number, low: string | null, high: string | null) =>
        low !== null && high !== null
          ? `mean of ${n} seeds · 95% CI: [${low}, ${high}]`
          : `mean of ${n} ${n === 1 ? "seed" : "seeds"} · no CI (a single seed)`,
      noCell:
        "A replicated comparison has no single value for this row: open the group to see each variant and the paired tests.",
    },
    mixedTasks: {
      title: "Different tasks in one comparison",
      body: "Metrics of different tasks can't be compared: a detection's mAP and a regression's R² don't measure the same thing or share a scale. Go back to the history and select runs of the same task.",
    },
    configDiffTitle: "// config diff (highlighted cells differ from the first run)",
    config: {
      architecture: "Architecture",
      numClasses: "Number of classes",
      pretrained: "Pretrained",
      task: "Task",
      learningRate: "Learning rate",
      optimizer: "Optimizer",
      batchSize: "Batch size",
      epochsMax: "Epochs (max)",
      weightDecay: "Weight decay",
      seed: "Seed",
      mixedPrecision: "Mixed precision",
      scheduler: "Scheduler",
      imageSize: "Image size",
      horizontalFlip: "Horizontal flip",
      rotation: "Rotation (°)",
      colorJitter: "Color jitter",
      preprocessing: "Preprocessing",
    },
    preprocessingTitle: "// preprocessing pipelines",
    noPreprocessing: "no preprocessing",
    curvePicker: "Curves",
    curveTitle: (label: string) => `${label} × epoch`,
    noCurves: (names: string, replicates: boolean) =>
      `No per-epoch curve, left out of the chart: ${names}.` +
      (replicates
        ? " A replicate set keeps the mean over its seeds, not each epoch's history: open one of its seeds to see the curve."
        : ""),
    curves: {
      train_loss: "Train loss",
      train_accuracy: "Train accuracy",
      val_loss: "Val loss",
      val_accuracy: "Val accuracy",
      map50_95: "Val mAP@50-95",
      map50: "Val mAP@50",
      precision: "Val precision",
      recall: "Val recall",
      box_loss: "Val box loss",
      val_box_loss: "Val box loss",
      train_box_loss: "Train box loss",
      train_cls_loss: "Train cls loss",
      train_dfl_loss: "Train DFL loss",
      val_cls_loss: "Val cls loss",
      val_dfl_loss: "Val DFL loss",
      val_r2: "Val R²",
      val_rmse: "Val RMSE",
      val_mae: "Val MAE",
      val_mse: "Val MSE",
      val_miou: "Val mIoU",
      val_dice: "Val Dice",
      val_pixel_acc: "Val pixel accuracy",
      val_auroc: "Val AUROC",
      val_image_f1: "Val image F1",
      val_threshold: "Val decision threshold",
    },
  },
  guidedTour: {
    dialogLabel: "VisionForge guide",
    closeLabel: "Close the guide",
    eyebrow: "First time here",
    inviteTitle: "Want a guide?",
    inviteBody:
      "There are two. The interface tour makes seven quick stops at the main spots: where to pick a task, how to point to your dataset, what's already set up for you and where your results are saved. The first training takes you through a real run on a sample dataset that VisionForge creates, and waits for you to do each step. You can leave at any time — both stay in the header's “guide” menu.",
    notNow: "Not now",
    next: "Next →",
    finish: "Done",
    // Under a step that waits for the researcher: what is still missing, and then that it is done.
    gateDone: "Done — you can continue.",
  },
  tour: {
    tabs: {
      title: "Pick a task",
      body: "Each tab is a complete task: classification, detection, regression, segmentation or anomaly detection. Switching tabs swaps the whole form, the metrics and the accent color — nothing carries over between tasks by accident.",
    },
    dataset: {
      title: "Point to your dataset",
      body: "Pick the root folder and VisionForge finds the train, validation and test subfolders by their usual names. If yours are named differently, the selectors next to it let you map them without renaming anything on disk.",
    },
    parameters: {
      title: "The parameters that matter come first",
      body: "Each panel shows the essentials first — epochs, batch size, learning rate — and tucks the rest under “Advanced”. The defaults were tuned for each task, so starting without changing anything is a perfectly good choice. The “i” next to each label explains what the field does.",
    },
    device: {
      title: "GPU or CPU",
      body: "VisionForge detects your hardware and uses the GPU when one is available. You can force the CPU here: it's slower, but it runs anywhere, and it helps tell whether an error comes from the code or from the GPU.",
    },
    train: {
      title: "Train",
      body: "This button runs whatever is selected — a single run, a grid search, cross-validation or replicates. While it runs, a live view shows the progress and each epoch's metrics; you can minimize it and come back to it later. It also has a Stop button: the run finishes the epoch in progress, starts nothing new and keeps what it has done so far — an interrupted fold, model, replicate or trial stays out of the mean and the ranking.",
    },
    history: {
      title: "Everything is saved",
      body: "Every run saves its configuration, per-epoch metrics, plots and weights to disk. From the history you can reopen a run, compare runs side by side, resume an interrupted run and test the model on new images.",
    },
    datasets: {
      title: "Your datasets",
      body: "Here you download a dataset (torchvision, Roboflow, Kaggle or Hugging Face) to a local folder and point any task at it. Once the folder is set, the task's panel shows the class distribution — worth a look before your first run: most odd results trace back to an imbalanced dataset.",
    },
  },
  // The guide menu and the guides' own texts (lib/guides/, ADR-115). The tour's
  // stops are `tour`; "First training" is `guides.firstTraining`.
  guides: {
    menuLabel: "Guides",
    tour: {
      title: "Interface tour",
      summary: "Seven stops through the main parts of the screen.",
    },
    firstTraining: {
      title: "First training",
      summary:
        "A real run, step by step, on a sample dataset of the task on the active tab.",
      action: "Create sample dataset",
      created: (path: string) => `Done, created at ${path}.`,
      existed: (path: string) =>
        `That folder already existed, so I will use it: ${path}.`,
      failed: (message: string) => `Could not create the dataset: ${message}`,
      datasetWaiting: "Fill in the dataset field to continue.",
      trainWaiting: "Click Train to continue.",
      resultWaiting: "Waiting for the training to finish…",
      history: {
        title: "Everything was saved",
        body: "The run is already in the History, under outputs/models/<name>/<date> (in your profile's folder, if you use one). The run.json holds the configuration and the metrics of every epoch; to see the effect of chance, run again with another seed and compare the two there.",
      },
      // Each task's script (ADR-117): what the data is, the parameters that
      // matter, what to watch while it trains and the result metric.
      tasks: {
        classification: {
          sample: {
            title: "Create a sample dataset",
            body: "You do not need images of your own for a first run: the button below creates a synthetic two-class dataset. It comes split into train, validation and test because each part has a job: the model learns on train, validation shows during training whether it is generalising, and test is used only at the end, on images the model has never seen.",
          },
          dataset: {
            title: "Point at the dataset",
            body: "The dataset field now points at the sample folder, and VisionForge has already found the train, val and test subfolders and the two classes (class_a and class_b), setting the model up for them. With a dataset of your own, this is where you pick the folder.",
          },
          parameters: {
            title: "The few parameters that matter",
            body: "Set epochs to 3: an epoch is one full pass over the training images, and three are enough for this dataset. Batch size (images per step) and learning rate (the size of each step) can stay as they are, and so can the rest. The first time, the model downloads the pretrained ResNet-18 weights (about 45 MB), which are cached afterwards.",
          },
          train: {
            title: "Train",
            body: "Click Train. If the machine has no GPU, pick CPU in the selector beside it: with this dataset the run takes a few seconds.",
          },
          watching: {
            title: "What is happening now",
            body: "The training sheet shows the current epoch and, after each one, the training loss and the validation loss. Loss is the model's error, and both should fall. The training one is measured during the epoch, while the model is still changing; the validation one at its end, on images the model did not learn from. The “How it works” tab, open next to the log, draws what training is doing, with the live numbers.",
          },
          result: {
            title: "The result",
            body: "When it finishes, open “View results” and compare the two losses: if the training loss keeps falling while the validation loss stalls or rises, the model is memorising the training set (overfitting). Accuracy near 100% here is on purpose, because this data is easy: it teaches the mechanics, not what to expect with real images.",
          },
          others: {
            title: "What about the other tasks?",
            body: "Detection: YOLO-format boxes, measured by mAP.\nSegmentation: per-pixel masks, measured by mIoU.\nRegression: a CSV with the value to predict, measured by R² and RMSE.\nAnomaly: only normal images in training, measured by AUROC.\nSynthetic data is easy on purpose: to know what to expect, download a real dataset from the download card in Datasets. To do the same on another task, open this guide with its tab active.",
          },
        },
        detection: {
          sample: {
            title: "Create a sample dataset",
            body: "Detection needs images and boxes. The button below creates eight synthetic 128 px images, each with a bright square in the middle, and a .txt file with the same name holding the box in YOLO format: class, centre x, centre y, width and height, all between 0 and 1. The images/ and labels/ folders have train and val; mAP is measured on validation.",
          },
          dataset: {
            title: "Point at the dataset",
            body: "The field now points at the sample folder, and VisionForge set the image size to 128 px and the number of classes to 1, the object class that the folder's classes.txt declares. The highlighted card counts images and boxes per split: four and four, one box per image. There is no test folder, and that is fine.",
          },
          parameters: {
            title: "The few parameters that matter",
            body: "Here, leave everything as it is. With few epochs YOLO barely leaves its warm-up and mAP stays near zero: the default 100 are what it needs, and on this dataset they take about a minute on CPU. The model is the pretrained YOLO11n, and the first time Ultralytics downloads its weights (about 5 MB).",
          },
          train: {
            title: "Train",
            body: "Click Train. If the machine has no GPU, pick CPU in the selector beside it: with this dataset the 100 epochs take about a minute.",
          },
          watching: {
            title: "What is happening now",
            body: "The training sheet shows the current epoch and, after each one, the validation mAP, precision, recall and losses. The box loss measures where the model puts the box; the class loss (cls), whether it gets right what is in it. In the first epochs mAP stays near zero and only rises after the warm-up: that is expected. The “How it works” tab, open next to the log, draws what training is doing, with the live numbers.",
          },
          result: {
            title: "The result",
            body: "When it finishes, open “View results”. mAP50-95 is the average precision of the detections over IoU thresholds (the overlap between the predicted and the true box) from 0.5 to 0.95. Near 1 here is on purpose, because the square is easy: it teaches the mechanics, not what to expect with real photos, where much lower values are normal.",
          },
          others: {
            title: "And with your own data?",
            body: "With a dataset of your own, the folder needs images/ and labels/ with train and val (or a data.yaml, like the ones Roboflow exports) and a classes.txt with the names. Synthetic data is easy on purpose: to know what to expect, download a real dataset from the download card in Datasets. To do the same on another task, open this guide with its tab active.",
          },
        },
        regression: {
          sample: {
            title: "Create a sample dataset",
            body: "Regression predicts a number from the image. The button below creates plain-colour synthetic images and one CSV manifest per split (train.csv, val.csv and test.csv) with two columns: image, the file name, and target, the value to predict, which here grows with the image's red. There are 12 training rows, 6 validation and 4 test.",
          },
          dataset: {
            title: "Point at the dataset",
            body: "The field now points at the sample folder, and VisionForge set the image size to 64 px. The CSV and column names (image and target) are already the form's defaults. The highlighted card checks the manifests: the rows of each split, whether the columns exist and whether the images are in the images/ folder.",
          },
          parameters: {
            title: "The few parameters that matter",
            body: "Set epochs to 30 and batch size to 4. With 12 training images, the default batch of 32 is a single step per epoch, and 30 epochs would be 30 steps, too few to learn from; with 4 there are 90. The loss (MSE) is the usual one for regression. The first time, the model downloads the pretrained ResNet-18 weights (about 45 MB), which are cached afterwards.",
          },
          train: {
            title: "Train",
            body: "Click Train. If the machine has no GPU, pick CPU in the selector beside it: with this dataset the run takes about a minute.",
          },
          watching: {
            title: "What is happening now",
            body: "The training sheet shows the current epoch and, after each one, the training loss and the validation metrics: RMSE and MAE, the typical error in the target's unit (lower is better), and R², which is 1 for a perfect model and 0 for one that always predicts the mean. R² swings from one epoch to the next and the result keeps the best of them; in the first ones it can be negative, because the model is still worse than the mean. The “How it works” tab, open next to the log, draws what training is doing, with the live numbers.",
          },
          result: {
            title: "The result",
            body: "When it finishes, open “View results”. Look at the validation R² and RMSE (6 images): an R² near 0.9 is a good fit, and this data is easy on purpose. The test R² rests on only 4 images and swings a lot from one run to the next: a low value there is largely small-sample noise, not necessarily a fault in the model.",
          },
          others: {
            title: "And with your own data?",
            body: "With your own data you need a folder with the images and one CSV per split, with the file-name column and one or more target columns (the names are in the Dataset card). Synthetic data is easy on purpose: to know what to expect, download a real dataset from the download card in Datasets. To do the same on another task, open this guide with its tab active.",
          },
        },
        segmentation: {
          sample: {
            title: "Create a sample dataset",
            body: "Segmentation classifies every pixel. The button below creates image and mask pairs: each split folder has images/ and masks/, with files of the same name. A mask is a grey-scale image whose pixels hold the class number: 0 is the background, and 1 and 2 are the two classes of a coloured rectangle. There are 6 training pairs, 3 validation and 2 test.",
          },
          dataset: {
            title: "Point at the dataset",
            body: "The field now points at the sample folder, and VisionForge set the image size to 64 px and the model's number of classes to 3: the values 0, 1 and 2 that the highlighted card found in the masks. That number has to cover every value in the masks, or the run stops. The card also checks that each image has its mask.",
          },
          parameters: {
            title: "The few parameters that matter",
            body: "Set epochs to 8 and batch size to 2: with 6 training images, the default batch of 8 is a single step per epoch, and a few epochs are not enough. The loss is cross-entropy, the usual one for segmentation, and the U-Net is trained from scratch, with no weights to download.",
          },
          train: {
            title: "Train",
            body: "Click Train. If the machine has no GPU, pick CPU in the selector beside it: with this dataset the run takes about a minute.",
          },
          watching: {
            title: "What is happening now",
            body: "The training sheet shows the current epoch and, after each one, the loss and the validation metrics: mIoU (the average overlap between the predicted and the true mask, per class), Dice (similar, more generous) and pixel accuracy. Since the background fills almost everything, pixel accuracy is the most optimistic of the three: trust mIoU. The “How it works” tab, open next to the log, draws what training is doing, with the live numbers.",
          },
          result: {
            title: "The result",
            body: "When it finishes, open “View results”: the validation and test mIoU and Dice. Near 1 here is on purpose, because the rectangles are easy: the result teaches the mechanics, not what to expect with real images, where much lower values are normal.",
          },
          others: {
            title: "And with your own data?",
            body: "With your own data, each split needs images/ and masks/ with files of the same name, and the mask holds the class number at every pixel (a value such as 255 can mark pixels to ignore). Synthetic data is easy on purpose: to know what to expect, download a real dataset from the download card in Datasets. To do the same on another task, open this guide with its tab active.",
          },
        },
        anomaly: {
          sample: {
            title: "Create a sample dataset",
            body: "Anomaly detection learns from normal images only. The button below creates 8 synthetic plain-texture images in train/good and, in test/, 4 normal ones (good) and 4 with a defect (defect), a white square the normal ones never have. The model never sees a defect during training: it is only tested on them.",
          },
          dataset: {
            title: "Point at the dataset",
            body: "The field now points at the sample folder, and VisionForge set the image size to 64 px. The folder names (train, test and good) are already the defaults. The highlighted card counts the normal training images and the test ones, telling the normal apart from the defective.",
          },
          parameters: {
            title: "The few parameters that matter",
            body: "Set epochs to 3. The method is the autoencoder: it learns to rebuild normal images, and an image with a defect is rebuilt badly, which gives it away. PatchCore, the other option, does not train in epochs: it stores features of the normal images and compares each new image with them. The rest can stay as it is.",
          },
          train: {
            title: "Train",
            body: "Click Train. If the machine has no GPU, pick CPU in the selector beside it: with this dataset the run takes about a minute.",
          },
          watching: {
            title: "What is happening now",
            body: "The training sheet shows the current epoch and, after each one, the reconstruction loss (the error when rebuilding a normal image, which should fall) and, measured on the test images, the AUROC and the image F1. PatchCore has no epochs: the sheet shows phases (extracting features, building the memory bank and scoring), each with its own progress. The “How it works” tab, open next to the log, draws what training is doing, with the live numbers.",
          },
          result: {
            title: "The result",
            body: "When it finishes, open “View results”. AUROC measures whether the model scores defective images higher than normal ones: 1 is perfect, 0.5 is luck. The image F1 also depends on the threshold, the score cut-off, computed on the training normals: it can be below 1 even with an AUROC of 1. Here the defects are easy on purpose.",
          },
          others: {
            title: "And with your own data?",
            body: "With your own data, the folder needs train/good, with normal images only, and test/ with a good subfolder and one per defect type (the names are in the Dataset card). Synthetic data is easy on purpose: to know what to expect, download a real dataset from the download card in Datasets. To do the same on another task, open this guide with its tab active.",
          },
        },
      },
    },
  },
};
