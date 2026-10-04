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
};
