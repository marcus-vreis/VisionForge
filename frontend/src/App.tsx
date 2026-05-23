import { useEffect, useState } from "react";
import { fetchSchema } from "./api/client";
import { BottomBar } from "./components/BottomBar";
import type { DeviceSelection } from "./components/DeviceSelector";
import { Header } from "./components/Header";
import { HistoryOverlay } from "./components/HistoryOverlay";
import { ParamPanel } from "./components/ParamPanel";
import { buildDefaults } from "./lib/schema-defaults";
import { ResultsView } from "./components/ResultsView";
import { TabBar } from "./components/TabBar";
import { TaskHero } from "./components/TaskHero";
import { TrainingOverlay } from "./components/TrainingOverlay";
import { Waves, Particles } from "./components/Waves";
import { useExperiment } from "./hooks/useExperiment";
import type { JsonSchema } from "./types/schema";
import { TASKS } from "./types/tasks";

export default function App() {
  const { status, result, error, validationErrors, progressEvents, submit, reset } =
    useExperiment();

  const [activeKey, setActiveKey] = useState("classification");
  const [device, setDevice] = useState<DeviceSelection>({
    kind: "cuda",
    gpu_ids: null,
  });
  const [showHistory, setShowHistory] = useState(false);
  const [historyCount, setHistoryCount] = useState(0);
  const [overlayVisible, setOverlayVisible] = useState(false);
  const [resultsVisible, setResultsVisible] = useState(false);
  const [schema, setSchema] = useState<JsonSchema | null>(null);
  const [formData, setFormData] = useState<Record<string, unknown>>({});
  const [pipelineSummary, setPipelineSummary] = useState<string[]>([]);
  const [blockKind, setBlockKind] = useState<string>("classification");
  const [queueSize, setQueueSize] = useState<number | undefined>(undefined);

  const showOverlay =
    overlayVisible &&
    (status.status === "running" ||
      status.status === "completed" ||
      status.status === "failed");

  useEffect(() => {
    fetchSchema()
      .then((s) => {
        setSchema(s);
        const defaults = buildDefaults(s, s.$defs ?? {}) as Record<
          string,
          unknown
        >;
        setFormData(defaults);
      })
      .catch(() => {
        /* server not running during build — ignore */
      });
  }, []);

  const handleTrain = async () => {
    if (activeKey !== "classification") return;
    reset();
    setResultsVisible(false);
    setOverlayVisible(true);
    // Inject the live device selection so the backend actually honors it
    // (instead of always defaulting to CUDA when present).
    const payload: Record<string, unknown> = {
      ...formData,
      device: { kind: device.kind, gpu_ids: device.gpu_ids },
    };
    // Extract pipeline filter names so the overlay can surface what's active.
    const data = (payload["data"] as Record<string, unknown> | undefined) ?? {};
    const pp = (data["preprocessing"] as Record<string, unknown> | undefined) ?? {};
    const steps = Array.isArray(pp["steps"])
      ? (pp["steps"] as Array<Record<string, unknown>>)
      : [];
    setPipelineSummary(steps.map((s) => String(s["kind"] ?? "")).filter(Boolean));

    // Surface the active block kind + queue size so the overlay can warn
    // about multi-trial blocks before the first SSE epoch lands.
    const kind = String(payload["block"] ?? "classification");
    setBlockKind(kind);
    let qSize: number | undefined;
    if (kind === "grid_search") {
      const gs = (payload["grid_search"] ?? {}) as Record<string, unknown>;
      const hp = (gs["hyperparameters"] ?? {}) as Record<string, unknown>;
      const trials = Object.values(hp).reduce<number>(
        (acc, vals) => acc * Math.max(Array.isArray(vals) ? vals.length : 1, 1),
        Object.keys(hp).length === 0 ? 0 : 1,
      );
      qSize = trials || undefined;
    } else if (kind === "random_search") {
      const rs = (payload["random_search"] ?? {}) as Record<string, unknown>;
      const n = rs["n_trials"];
      qSize = typeof n === "number" ? n : undefined;
    } else if (kind === "cross_validation") {
      const cv = (payload["cross_validation"] ?? {}) as Record<string, unknown>;
      const n = cv["n_folds"];
      qSize = typeof n === "number" ? n : undefined;
    } else if (kind === "model_comparison") {
      const mc = (payload["model_comparison"] ?? {}) as Record<string, unknown>;
      const names = mc["model_names"];
      qSize = Array.isArray(names) ? names.length : undefined;
    }
    setQueueSize(qSize);

    await submit(payload);
  };

  const activeTask = TASKS.find((t) => t.key === activeKey) ?? TASKS[0];
  const showResults = resultsVisible && result !== null;

  return (
    <div
      className="stage"
      data-task={activeKey}
      style={{
        minHeight: "100vh",
        position: "relative",
        overflow: "hidden",
        background:
          `radial-gradient(1100px 600px at 80% -10%, oklch(0.20 0.04 260 / 0.30), transparent 60%),` +
          `radial-gradient(900px 500px at -10% 110%, var(--accent-soft), transparent 60%),` +
          `var(--vf-bg)`,
        transition: "background 600ms ease",
        fontFamily: "var(--font-sans)",
        color: "var(--vf-text)",
      }}
    >
      <Waves />
      <Particles />

      <Header />

      <TabBar tasks={TASKS} activeKey={activeKey} setActiveKey={setActiveKey} />

      <main
        style={{
          position: "relative",
          zIndex: 2,
          maxWidth: 1280,
          margin: "0 auto",
          padding: "34px 40px 140px",
        }}
      >
        <TaskHero task={activeTask} />

        {showResults ? (
          <ResultsView
            result={result}
            taskAccent={activeTask.accent}
            onClose={() => {
              setResultsVisible(false);
              reset();
            }}
          />
        ) : (
          <ParamPanel
            task={activeTask}
            schema={schema}
            formData={formData}
            setFormData={setFormData}
            validationErrors={validationErrors}
          />
        )}

        {error && !showOverlay && (
          <div
            style={{
              marginTop: 16,
              padding: "14px 18px",
              background: "oklch(0.704 0.191 22.216 / 0.10)",
              border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
              borderRadius: 12,
              fontFamily: "var(--font-mono)",
              fontSize: 13,
              color: "oklch(0.85 0.14 22)",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              lineHeight: 1.55,
            }}
          >
            <div style={{ fontSize: 10, letterSpacing: "0.16em", textTransform: "uppercase", color: "oklch(0.7 0.18 22)", marginBottom: 4 }}>
              Erro
            </div>
            {error}
          </div>
        )}
      </main>

      <BottomBar
        onHistory={() => setShowHistory(true)}
        onTrain={() => void handleTrain()}
        disabled={status.status === "running" || activeKey !== "classification"}
        historyCount={historyCount}
        selection={device}
        onSelectionChange={setDevice}
        isRunning={status.status === "running"}
        trainingMinimized={status.status === "running" && !overlayVisible}
        onReopenTraining={() => setOverlayVisible(true)}
      />

      <HistoryOverlay
        open={showHistory}
        onClose={() => setShowHistory(false)}
        onCountChange={setHistoryCount}
      />

      {showOverlay && (
        <TrainingOverlay
          status={status}
          progressEvents={progressEvents}
          taskAccent={activeTask.accent}
          taskLabel={activeTask.label}
          pipelineSummary={pipelineSummary}
          blockKind={blockKind}
          queueSize={queueSize}
          onClose={() => setOverlayVisible(false)}
          onViewResults={() => {
            setOverlayVisible(false);
            setResultsVisible(true);
          }}
        />
      )}
    </div>
  );
}
