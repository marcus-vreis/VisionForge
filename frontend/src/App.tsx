import { useCallback, useEffect, useState } from "react";
import {
  fetchSchema,
  fetchTasks,
  runCustomReplicates,
  runCustomSweep,
  runCustomTask,
  runReplicates,
  runSweep,
  runTaskCv,
} from "./api/client";
import type { CvPayload } from "./components/CvCard";
import type { PanelStrategy } from "./components/ExperimentHeader";
import type { SweepPayload } from "./components/SweepCard";
import type { ReplicatesPayload } from "./lib/replicates-form";
import { BottomBar } from "./components/BottomBar";
import type { DeviceSelection } from "./components/DeviceSelector";
import { Header } from "./components/Header";
import { DatasetsOverlay } from "./components/DatasetsOverlay";
import { HistoryOverlay } from "./components/HistoryOverlay";
import { ParamPanel } from "./components/ParamPanel";
import { DetectionPanel } from "./components/DetectionPanel";
import { RegressionPanel } from "./components/RegressionPanel";
import { SegmentationPanel } from "./components/SegmentationPanel";
import { AnomalyPanel } from "./components/AnomalyPanel";
import {
  buildDetectionDataPayload,
  buildDetectionTrainingPayload,
  makeDefaultDetectionForm,
  type DetectionForm,
} from "./lib/detection-models";
import {
  buildRegressionPayload,
  makeDefaultRegressionForm,
  type RegressionForm,
} from "./lib/regression-models";
import {
  buildSegmentationPayload,
  makeDefaultSegmentationForm,
  type SegmentationForm,
} from "./lib/segmentation-models";
import {
  buildAnomalyPayload,
  makeDefaultAnomalyForm,
  type AnomalyForm,
} from "./lib/anomaly-models";
import { buildDefaults } from "./lib/schema-defaults";
import {
  buildCustomPayload,
  isCustomTask,
  mergeTasks,
} from "./lib/custom-tasks";
import { CustomTaskPanel } from "./components/CustomTaskPanel";
import { ResultsView } from "./components/ResultsView";
import { TabBar } from "./components/TabBar";
import { TaskHero } from "./components/TaskHero";
import { TrainingOverlay } from "./components/TrainingOverlay";
import { useExperiment } from "./hooks/useExperiment";
import type { RunResponse } from "./types/run";
import type { JsonSchema } from "./types/schema";
import { TASKS, type TaskDefinition } from "./types/tasks";

/** Standalone tasks that expose the comparison/sweep advanced surface. */
type AdvancedTask = "regression" | "segmentation" | "detection" | "anomaly";

/** The main button says what it will actually run (ADR-059 follow-up). */
const TRAIN_LABELS: Record<PanelStrategy, string> = {
  simple: "▶ Treinar",
  cv: "▶ Rodar K-fold",
  sweep: "▶ Rodar sweep",
  replicates: "▶ Rodar réplicas",
};

export default function App() {
  const { status, result, error, validationErrors, progressEvents, submit, reset } =
    useExperiment();

  const [activeKey, setActiveKey] = useState("classification");
  const [device, setDevice] = useState<DeviceSelection>({
    kind: "cuda",
    gpu_ids: null,
  });
  const [showHistory, setShowHistory] = useState(false);
  const [showDatasets, setShowDatasets] = useState(false);
  const [historyCount, setHistoryCount] = useState(0);
  const [overlayVisible, setOverlayVisible] = useState(false);
  const [resultsVisible, setResultsVisible] = useState(false);
  const [schema, setSchema] = useState<JsonSchema | null>(null);
  const [formData, setFormData] = useState<Record<string, unknown>>({});
  const [pipelineSummary, setPipelineSummary] = useState<string[]>([]);
  const [blockKind, setBlockKind] = useState<string>("classification");
  const [queueSize, setQueueSize] = useState<number | undefined>(undefined);
  const [detectionForm, setDetectionForm] = useState<DetectionForm>(
    makeDefaultDetectionForm,
  );
  const [regressionForm, setRegressionForm] = useState<RegressionForm>(
    makeDefaultRegressionForm,
  );
  const [segmentationForm, setSegmentationForm] = useState<SegmentationForm>(
    makeDefaultSegmentationForm,
  );
  const [anomalyForm, setAnomalyForm] = useState<AnomalyForm>(
    makeDefaultAnomalyForm,
  );
  // Tabs are data-driven: built-ins are local, custom tasks arrive from
  // /api/tasks (ADR-058 brick 6). One form per custom key so switching tabs
  // preserves what the researcher typed.
  const [tasks, setTasks] = useState<TaskDefinition[]>(TASKS);
  const [customForms, setCustomForms] = useState<
    Record<string, Record<string, unknown>>
  >({});
  // The strategy lives in each panel's header; App mirrors it so the main
  // Treinar button runs what the researcher selected instead of silently
  // starting a plain single run. `runSignal` is the trigger the active
  // strategy's card listens to.
  const [strategyByTask, setStrategyByTask] = useState<
    Record<string, PanelStrategy>
  >({});
  const [runSignal, setRunSignal] = useState(0);

  // The overlay stays MOUNTED for the whole life of a run (hidden via CSS when
  // minimized) so its logs and progress survive minimize/reopen.
  const runActive =
    status.status === "running" ||
    status.status === "completed" ||
    status.status === "failed";
  const showOverlay = overlayVisible && runActive;
  const activeTask = tasks.find((t) => t.key === activeKey) ?? tasks[0];

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

  useEffect(() => {
    fetchTasks()
      .then((res) => setTasks(mergeTasks(TASKS, res.tasks)))
      .catch(() => {
        /* older server or none: the five built-in tabs still work */
      });
  }, []);

  const activeCustomForm = customForms[activeKey] ?? {};
  // Stable per key: CustomTaskPanel's schema effect depends on this identity.
  const setActiveCustomForm = useCallback(
    (next: Record<string, unknown>) =>
      setCustomForms((prev) => ({ ...prev, [activeKey]: next })),
    [activeKey],
  );

  // A custom task shares the whole single-run surface (overlay, results,
  // history); only the submit URL differs (ADR-058).
  const startCustom = async (
    key: string,
    body: Record<string, unknown>,
    kind: string,
    queue?: number,
    run?: (p: Record<string, unknown>) => Promise<RunResponse>,
  ) => {
    reset();
    setResultsVisible(false);
    setOverlayVisible(true);
    setPipelineSummary([]);
    setBlockKind(kind);
    setQueueSize(queue);
    await submit(body, { run: run ?? ((p) => runCustomTask(key, p)) });
  };

  const activeStrategy = strategyByTask[activeKey] ?? "simple";
  const setActiveStrategy = (s: PanelStrategy) =>
    setStrategyByTask((prev) => ({ ...prev, [activeKey]: s }));

  const handleTrain = async () => {
    // Classification encodes its strategy in config.block, so its payload
    // already carries it; the standalone panels keep theirs in cards, which
    // this signal triggers.
    if (activeStrategy !== "simple" && activeKey !== "classification") {
      setRunSignal((n) => n + 1);
      return;
    }
    if (isCustomTask(activeTask)) {
      await startCustom(
        activeTask.key,
        buildCustomPayload(activeCustomForm, device),
        "custom",
      );
      return;
    }
    if (activeKey === "detection") {
      reset();
      setResultsVisible(false);
      setOverlayVisible(true);
      setPipelineSummary([]);
      setBlockKind("detection");
      setQueueSize(undefined);
      const payload: Record<string, unknown> = {
        ...detectionForm,
        data: buildDetectionDataPayload(detectionForm.data),
        training: buildDetectionTrainingPayload(detectionForm.training),
        device: { kind: device.kind, gpu_ids: device.gpu_ids },
      };
      await submit(payload, { detection: true });
      return;
    }
    if (activeKey === "regression") {
      reset();
      setResultsVisible(false);
      setOverlayVisible(true);
      setPipelineSummary([]);
      setBlockKind("regression");
      setQueueSize(undefined);
      const payload: Record<string, unknown> = {
        ...buildRegressionPayload(regressionForm),
        device: { kind: device.kind, gpu_ids: device.gpu_ids },
      };
      await submit(payload, { regression: true });
      return;
    }
    if (activeKey === "segmentation") {
      reset();
      setResultsVisible(false);
      setOverlayVisible(true);
      setPipelineSummary([]);
      setBlockKind("segmentation");
      setQueueSize(undefined);
      const payload: Record<string, unknown> = {
        ...buildSegmentationPayload(segmentationForm),
        device: { kind: device.kind, gpu_ids: device.gpu_ids },
      };
      await submit(payload, { segmentation: true });
      return;
    }
    if (activeKey === "anomaly") {
      reset();
      setResultsVisible(false);
      setOverlayVisible(true);
      setPipelineSummary([]);
      setBlockKind("anomaly");
      setQueueSize(undefined);
      const payload: Record<string, unknown> = {
        ...buildAnomalyPayload(anomalyForm),
        device: { kind: device.kind, gpu_ids: device.gpu_ids },
      };
      await submit(payload, { anomaly: true });
      return;
    }
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

  // Model comparison (ADR-044) for the standalone tasks: trains the picked
  // architectures on the same dataset and ranks them. Reuses the overlay (queue
  // banner) + ResultsView (comparison report); no per-epoch stream.
  // Build the base task config dict for an advanced run (comparison / sweep).
  const buildTaskBase = (task: AdvancedTask): Record<string, unknown> => {
    if (task === "regression") return buildRegressionPayload(regressionForm);
    if (task === "segmentation") return buildSegmentationPayload(segmentationForm);
    if (task === "anomaly") return buildAnomalyPayload(anomalyForm);
    return {
      ...detectionForm,
      data: buildDetectionDataPayload(detectionForm.data),
      training: buildDetectionTrainingPayload(detectionForm.training),
    };
  };

  // Task K-fold CV (ADR-050): folds over the train split, fold-a-fold metrics
  // + mean ± std. Same overlay/results flow; one fold trains at a time.
  const handleTaskCv = async (
    task: "regression" | "segmentation",
    payload: CvPayload,
  ) => {
    reset();
    setResultsVisible(false);
    setOverlayVisible(true);
    setPipelineSummary([]);
    setBlockKind("cross_validation");
    setQueueSize(payload.n_folds);
    const config = {
      ...buildTaskBase(task),
      device: { kind: device.kind, gpu_ids: device.gpu_ids },
    };
    await submit({ config, ...payload }, { run: (p) => runTaskCv(task, p) });
  };

  // Multi-seed replicates (ADR-056): same config, N seeds, mean ± CI report.
  // Same overlay/results flow as sweeps; one trial trains at a time.
  const handleReplicates = async (
    task: AdvancedTask,
    payload: ReplicatesPayload,
  ) => {
    reset();
    setResultsVisible(false);
    setOverlayVisible(true);
    setPipelineSummary([]);
    setBlockKind("replicates");
    setQueueSize(payload.seeds?.length ?? payload.n_replicates ?? undefined);
    const config = {
      ...buildTaskBase(task),
      device: { kind: device.kind, gpu_ids: device.gpu_ids },
    };
    await submit({ config, ...payload }, { run: (p) => runReplicates(task, p) });
  };

  // Hyperparameter sweep (ADR-045) for the standalone tasks: grid/random search
  // over dot-paths, ranked by the chosen metric. Same overlay/results flow.
  const handleSweep = async (
    task: AdvancedTask,
    payload: SweepPayload,
  ) => {
    reset();
    setResultsVisible(false);
    setOverlayVisible(true);
    setPipelineSummary([]);
    setBlockKind(payload.mode === "grid" ? "grid_search" : "random_search");
    const space = payload.search_space;
    const qSize =
      payload.mode === "grid"
        ? Object.values(space).reduce<number>(
            (acc, v) => acc * (Array.isArray(v) ? v.length : 1),
            Object.keys(space).length === 0 ? 0 : 1,
          )
        : payload.n_trials;
    setQueueSize(qSize || undefined);
    const config = {
      ...buildTaskBase(task),
      device: { kind: device.kind, gpu_ids: device.gpu_ids },
    };
    await submit({ config, ...payload }, { run: (p) => runSweep(task, p) });
  };

  const showResults = resultsVisible && result !== null;

  return (
    <div
      className="stage"
      data-task={activeKey}
      style={{
        minHeight: "100vh",
        position: "relative",
        overflow: "hidden",
        fontFamily: "var(--font-sans)",
        color: "var(--vf-text)",
      }}
    >
      <Header />

      <TabBar tasks={tasks} activeKey={activeKey} setActiveKey={setActiveKey} />

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
        ) : isCustomTask(activeTask) ? (
          <CustomTaskPanel
            task={activeTask}
            formData={activeCustomForm}
            setFormData={setActiveCustomForm}
            validationErrors={validationErrors}
            busy={status.status === "running"}
            onSweep={(payload) =>
              void startCustom(
                activeTask.key,
                {
                  config: buildCustomPayload(activeCustomForm, device),
                  ...payload,
                },
                payload.mode === "grid" ? "grid_search" : "random_search",
                undefined,
                (p) => runCustomSweep(activeTask.key, p),
              )
            }
            onReplicates={(payload) =>
              void startCustom(
                activeTask.key,
                {
                  config: buildCustomPayload(activeCustomForm, device),
                  ...payload,
                },
                "replicates",
                payload.seeds?.length ?? payload.n_replicates ?? undefined,
                (p) => runCustomReplicates(activeTask.key, p),
              )
            }
            onStrategyChange={setActiveStrategy}
            runSignal={runSignal}
          />
        ) : activeKey === "detection" ? (
          <DetectionPanel
            formData={detectionForm}
            setFormData={setDetectionForm}
            accent={activeTask.accent}
            validationErrors={validationErrors}
            busy={status.status === "running"}
            onSweep={(payload) => void handleSweep("detection", payload)}
            onReplicates={(payload) =>
              void handleReplicates("detection", payload)
            }
            onStrategyChange={setActiveStrategy}
            runSignal={runSignal}
          />
        ) : activeKey === "regression" ? (
          <RegressionPanel
            formData={regressionForm}
            setFormData={setRegressionForm}
            accent={activeTask.accent}
            validationErrors={validationErrors}
            busy={status.status === "running"}
            onSweep={(payload) => void handleSweep("regression", payload)}
            onReplicates={(payload) =>
              void handleReplicates("regression", payload)
            }
            onCv={(payload) => void handleTaskCv("regression", payload)}
            onStrategyChange={setActiveStrategy}
            runSignal={runSignal}
          />
        ) : activeKey === "segmentation" ? (
          <SegmentationPanel
            formData={segmentationForm}
            setFormData={setSegmentationForm}
            accent={activeTask.accent}
            validationErrors={validationErrors}
            busy={status.status === "running"}
            onSweep={(payload) => void handleSweep("segmentation", payload)}
            onReplicates={(payload) =>
              void handleReplicates("segmentation", payload)
            }
            onCv={(payload) => void handleTaskCv("segmentation", payload)}
            onStrategyChange={setActiveStrategy}
            runSignal={runSignal}
          />
        ) : activeKey === "anomaly" ? (
          <AnomalyPanel
            formData={anomalyForm}
            setFormData={setAnomalyForm}
            accent={activeTask.accent}
            validationErrors={validationErrors}
            busy={status.status === "running"}
            onSweep={(payload) => void handleSweep("anomaly", payload)}
            onReplicates={(payload) =>
              void handleReplicates("anomaly", payload)
            }
            onStrategyChange={setActiveStrategy}
            runSignal={runSignal}
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
        onDatasets={() => setShowDatasets(true)}
        onTrain={() => void handleTrain()}
        disabled={status.status === "running"}
        trainLabel={TRAIN_LABELS[activeStrategy] ?? "▶ Treinar"}
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
        // `activeKey` is already the family the history groups by: the task
        // tabs are classification/detection/... and custom tasks carry their
        // own key, which the history prefixes to match its `custom:<key>` runs.
        initialTask={
          isCustomTask(activeTask) ? `custom:${activeKey}` : activeKey
        }
      />

      <DatasetsOverlay
        open={showDatasets}
        onClose={() => setShowDatasets(false)}
      />

      {runActive && (
        <TrainingOverlay
          status={status}
          progressEvents={progressEvents}
          visible={overlayVisible}
          taskAccent={activeTask.accent}
          taskLabel={activeTask.label}
          taskKey={activeKey}
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
