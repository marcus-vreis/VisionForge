import { useEffect, useRef, useState } from "react";
import type { RunStatus, TrainingEvent } from "../types/run";

interface TrainingOverlayProps {
  status: RunStatus;
  progressEvents: TrainingEvent[];
  /** When false the overlay is hidden with CSS but stays MOUNTED, so logs and
   *  progress survive minimize/reopen instead of resetting. */
  visible: boolean;
  taskAccent: string;
  taskLabel: string;
  /** Active task key (classification/detection/…) — drives the CLI echo line. */
  taskKey: string;
  /** Ordered preprocessing filter names from the submitted config, if any. */
  pipelineSummary?: string[];
  /** Block key from config — drives the "multi-trial" banner. Empty/unknown → no banner. */
  blockKind?: string;
  /** For grid_search: total trials (cartesian product); for random_search: n_trials.
   *  Lets the overlay say "fila: N treinos" before the first SSE epoch arrives. */
  queueSize?: number;
  onClose: () => void;
  onViewResults?: () => void;
}

const QUEUE_BLOCKS: Record<string, string> = {
  grid_search: "Grid search",
  random_search: "Random search",
  model_comparison: "Comparação de modelos",
  cross_validation: "K-Fold CV",
  replicates: "Réplicas multi-seed",
};

/** Modal overlay shown while an experiment is running or just completed. */
export function TrainingOverlay({
  status,
  progressEvents,
  visible,
  taskAccent,
  taskLabel,
  taskKey,
  pipelineSummary,
  blockKind,
  queueSize,
  onClose,
  onViewResults,
}: TrainingOverlayProps) {
  const queueLabel = blockKind ? QUEUE_BLOCKS[blockKind] : undefined;
  const isRunning = status.status === "running";
  const isCompleted = status.status === "completed";
  const hasFailed = status.status === "failed";
  const isFinished = isCompleted || hasFailed;
  // Waiting for the GPU (ADR-075): no epochs will arrive yet, so the synthetic
  // progress crawl must stay off — a moving bar here would claim work that has
  // not begun.
  const isQueued = status.status === "queued";

  // Derive real progress from the most recent epoch_end event.
  const epochEvents = progressEvents.filter(
    (e): e is Extract<TrainingEvent, { event: "epoch_end" }> =>
      e.event === "epoch_end",
  );
  const startEvent = progressEvents.find(
    (e): e is Extract<TrainingEvent, { event: "start" }> => e.event === "start",
  );
  const latestEpoch = epochEvents.at(-1);
  const totalEpochs = latestEpoch?.total_epochs ?? startEvent?.total_epochs ?? 0;
  const currentEpoch = latestEpoch?.epoch ?? 0;

  // Multi-trial blocks (grid/random search) stream a trial_start before each
  // inner training and a trial_end after. Progress then spans all trials, not
  // just the active one's epochs.
  const trialStartEvents = progressEvents.filter(
    (e): e is Extract<TrainingEvent, { event: "trial_start" }> =>
      e.event === "trial_start",
  );
  const trialEndCount = progressEvents.filter(
    (e) => e.event === "trial_end",
  ).length;
  const latestTrialStart = trialStartEvents.at(-1);
  const totalTrials =
    latestEpoch?.total_trials ?? latestTrialStart?.total_trials ?? 0;
  const isMultiTrial = totalTrials > 1;
  const currentTrialIndex = latestTrialStart?.trial_index ?? 0;
  // Don't double-count: the just-finished trial's epochs are already folded
  // into trialEndCount, so only add an in-flight fraction for a live trial.
  const currentTrialDone = trialEndCount > currentTrialIndex;
  const epochFraction =
    !currentTrialDone &&
    latestEpoch &&
    (latestEpoch.trial_index ?? 0) === currentTrialIndex &&
    totalEpochs > 0
      ? currentEpoch / totalEpochs
      : 0;

  // Fall back to synthetic crawl when SSE has not delivered an epoch yet.
  const [fakeProgress, setFakeProgress] = useState(0);
  const hasRealProgress = currentEpoch > 0 || trialEndCount > 0;
  const realProgress = isMultiTrial
    ? Math.min(1, (trialEndCount + epochFraction) / totalTrials)
    : totalEpochs > 0
      ? currentEpoch / totalEpochs
      : 0;

  const [logs, setLogs] = useState<string[]>([
    `$ visionforge train --task ${taskKey}`,
    `> inicializando runtime · ${status.run_id ?? "..."}`,
    `> carregando dataset…`,
  ]);
  const logRef = useRef<HTMLDivElement>(null);
  // Tracks which terminal status we already processed so the effect only runs once.
  const handledStatusRef = useRef<string | null>(null);

  // Synthetic progress crawl only while running and before real epochs arrive.
  useEffect(() => {
    if (!isRunning || hasRealProgress) return;
    const id = setInterval(() => {
      setFakeProgress((p) => Math.min(p + 0.004, 0.92));
    }, 2000);
    return () => clearInterval(id);
  }, [isRunning, hasRealProgress]);

  // Emit a separator line when a new trial starts (grid/random search).
  useEffect(() => {
    if (latestTrialStart === undefined) return;
    const { trial_index, total_trials, overrides } = latestTrialStart;
    const ov = Object.entries(overrides ?? {})
      .map(([k, v]) => `${k.split(".").at(-1)}=${v}`)
      .join(" · ");
    const line =
      `── trial ${trial_index + 1}/${total_trials}` +
      (ov ? ` · ${ov}` : "") +
      " ──";
    // Defer the append so it does not run synchronously inside the effect body
    // (avoids cascading renders); same pattern as the terminal-state effect.
    const timer = setTimeout(() => {
      setLogs((prev) => {
        if (prev.at(-1) === line) return prev;
        return [...prev.slice(-24), line];
      });
    }, 0);
    return () => clearTimeout(timer);
  }, [latestTrialStart]);

  // Append a log line for each new epoch_end event.
  useEffect(() => {
    if (latestEpoch === undefined) return;
    const { epoch, total_epochs, train_loss, val_loss, val_accuracy } =
      latestEpoch;
    const trialTag =
      latestEpoch.trial_index !== undefined && latestEpoch.total_trials
        ? `[t${latestEpoch.trial_index + 1}/${latestEpoch.total_trials}] `
        : "";
    // Detection runs carry mAP / precision / per-component losses; show those
    // instead of the classification loss/acc triple. `map50` is present (maybe
    // null) only on detection events, so it discriminates the task here.
    const isDetection = latestEpoch.map50 !== undefined;
    const fmt = (v: number | null | undefined): string =>
      v === null || v === undefined ? "—" : v.toFixed(4);
    // Researcher-defined tasks (ADR-058) stream their own declared metrics as
    // val_<name> and have no val_loss — list what the event actually carries
    // instead of assuming the classification triple (which crashed the
    // overlay when a custom task's first epoch arrived).
    const customMetrics = Object.entries(latestEpoch)
      .filter(
        ([k, v]) =>
          k.startsWith("val_") &&
          k !== "val_accuracy" &&
          k !== "val_loss" &&
          !k.endsWith("_loss") &&
          typeof v === "number",
      )
      .map(([k, v]) => `${k.slice(4)}=${fmt(v as number)}`);

    const metricsPart = isDetection
      ? [
          `map50=${fmt(latestEpoch.map50)}`,
          `map50-95=${fmt(latestEpoch.map50_95)}`,
          `P=${fmt(latestEpoch.precision)}`,
          `R=${fmt(latestEpoch.recall)}`,
          `box=${fmt(latestEpoch.val_box_loss ?? latestEpoch.box_loss)}`,
          `cls=${fmt(latestEpoch.val_cls_loss)}`,
          `dfl=${fmt(latestEpoch.val_dfl_loss)}`,
        ].join(" · ")
      : customMetrics.length > 0
        ? [`loss=${fmt(train_loss)}`, ...customMetrics].join(" · ")
        : `loss=${fmt(train_loss)} · val_loss=${fmt(val_loss)} · val_acc=${fmt(val_accuracy)}`;
    const line = `> ${trialTag}epoch ${epoch}/${total_epochs} · ${metricsPart}`;
    // Defer the append (see the trial-separator effect above).
    const timer = setTimeout(() => {
      setLogs((prev) => {
        // Avoid duplicate lines if the effect runs twice in strict-mode.
        if (prev.at(-1) === line) return prev;
        return [...prev.slice(-24), line];
      });
    }, 0);
    return () => clearTimeout(timer);
  }, [latestEpoch]);

  // Handle terminal states — subscribe to status.status changes as an external signal.
  useEffect(() => {
    if (handledStatusRef.current === status.status) return;
    if (isCompleted) {
      handledStatusRef.current = status.status;
      const timer = setTimeout(() => {
        setLogs((prev) => [...prev.slice(-24), "$ training complete"]);
      }, 0);
      return () => clearTimeout(timer);
    }
    if (hasFailed) {
      handledStatusRef.current = status.status;
      const msg = status.error ?? "erro desconhecido";
      const timer = setTimeout(() => {
        setLogs((prev) => [
          ...prev.slice(-24),
          `$ training failed · ${msg}`,
        ]);
      }, 0);
      return () => clearTimeout(timer);
    }
  }, [status.status, isCompleted, hasFailed, status.error]);

  // Auto-scroll logs — also when the overlay is re-shown after a minimize
  // (scrollHeight is 0 while display:none, so re-anchor on visibility).
  useEffect(() => {
    if (visible && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs, visible]);

  const progressFraction = isCompleted
    ? 1
    : hasRealProgress
      ? realProgress
      : fakeProgress;
  const pct = Math.round(progressFraction * 100);

  return (
    <div
      onClick={(e) => {
        if (e.target === e.currentTarget && isFinished) onClose();
      }}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 90,
        background: "rgba(4,5,7,0.78)",
        display: visible ? "flex" : "none",
        alignItems: "center",
        justifyContent: "center",
        animation: "overlayIn 220ms ease forwards",
        padding: 24,
      }}
    >
      <div
        style={{
          width: "min(720px, 100%)",
          background: "rgba(12,14,18,0.95)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 18,
          padding: 28,
          animation: "sheetIn 260ms cubic-bezier(0.2, 0.9, 0.2, 1) forwards",
        }}
      >
        {/* Header */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
            marginBottom: 18,
          }}
        >
          <span
            style={{
              width: 16,
              height: 16,
              borderRadius: "50%",
              border: `2px solid ${taskAccent}`,
              borderTopColor: isFinished ? taskAccent : "transparent",
              animation: isFinished ? "none" : "spin 0.8s linear infinite",
              flexShrink: 0,
            }}
          />
          <div style={{ flex: 1 }}>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                letterSpacing: "0.22em",
                textTransform: "uppercase",
                color: "var(--vf-text-muted)",
              }}
            >
              {isFinished
                ? hasFailed
                  ? "training failed"
                  : "training complete"
                : isQueued
                  ? `na fila · ${taskLabel}`
                  : `training · ${taskLabel}`}
            </div>
            <div
              style={{
                fontSize: 18,
                fontWeight: 600,
                marginTop: 4,
                fontFamily: "var(--font-mono)",
                color: "var(--vf-text)",
              }}
            >
              {status.run_id ?? "iniciando…"}
            </div>
            {isQueued && (
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  marginTop: 6,
                  color: "var(--vf-text-muted)",
                }}
              >
                {status.position
                  ? `aguardando a GPU — ${status.position}º na fila`
                  : "aguardando a GPU"}
                {typeof status.queued === "number" && status.queued > 1
                  ? ` · ${status.queued} submissões esperando`
                  : ""}
                . O treino começa sozinho quando chegar a vez.
              </div>
            )}
          </div>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 28,
              fontWeight: 600,
              color: hasFailed ? "oklch(0.74 0.18 22)" : taskAccent,
            }}
          >
            {pct}%
          </div>
        </div>

        {/* Progress bar */}
        <div
          style={{
            height: 6,
            borderRadius: 999,
            overflow: "hidden",
            background: "rgba(255,255,255,0.04)",
            marginBottom: 18,
          }}
        >
          <div
            style={{
              width: `${pct}%`,
              height: "100%",
              background: hasFailed
                ? `linear-gradient(90deg, oklch(0.74 0.18 22), oklch(0.74 0.18 22 / 0.6))`
                : `linear-gradient(90deg, ${taskAccent}, ${taskAccent}aa)`,
              boxShadow: `0 0 12px ${taskAccent}`,
              transition: "width 320ms ease",
            }}
          />
        </div>

        {/* Failure detail panel — shown above logs when failed */}
        {hasFailed && (
          <div
            style={{
              padding: "14px 16px",
              marginBottom: 14,
              background: "oklch(0.704 0.191 22.216 / 0.10)",
              border: "1px solid oklch(0.704 0.191 22.216 / 0.45)",
              borderRadius: 12,
              fontFamily: "var(--font-mono)",
              fontSize: 12.5,
              color: "oklch(0.88 0.14 22)",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              lineHeight: 1.55,
              maxHeight: 180,
              overflowY: "auto",
            }}
          >
            <div
              style={{
                fontSize: 10,
                letterSpacing: "0.18em",
                textTransform: "uppercase",
                color: "oklch(0.7 0.18 22)",
                marginBottom: 6,
              }}
            >
              Detalhe do erro
            </div>
            {status.error ?? "Erro desconhecido — verifique os logs do servidor."}
          </div>
        )}

        {/* Multi-trial queue banner — when the block runs N inner trainings
            sequentially (grid_search, random_search, etc.) the overlay's
            single progress bar tracks only the active trial. This banner
            sets expectations up front. */}
        {queueLabel && (
          <div
            style={{
              marginBottom: 14,
              padding: "10px 14px",
              background: "rgba(180, 140, 255, 0.10)",
              border: "1px solid rgba(180, 140, 255, 0.45)",
              borderRadius: 12,
              display: "flex",
              flexDirection: "column",
              gap: 4,
            }}
          >
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 9,
                letterSpacing: "0.18em",
                textTransform: "uppercase",
                color: "rgba(220, 190, 255, 0.85)",
              }}
            >
              ⛓ fila de treinos · {queueLabel}
              {queueSize && queueSize > 1 ? ` · ${queueSize} runs` : ""}
            </div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--vf-text-dim)",
                lineHeight: 1.5,
              }}
            >
              Este bloco executa múltiplos treinos sequencialmente. A barra de
              progresso reflete o trial corrente; o resultado agregado aparece
              em "Ver resultados" ao final.
            </div>
          </div>
        )}

        {/* Active preprocessing pipeline — surfaced so the researcher can
            confirm at a glance which filters this run is using. */}
        {pipelineSummary && pipelineSummary.length > 0 && (
          <div
            style={{
              marginBottom: 14,
              padding: "10px 14px",
              background: "oklch(0.72 0.16 150 / 0.10)",
              border: "1px solid oklch(0.72 0.16 150 / 0.40)",
              borderRadius: 12,
              display: "flex",
              flexDirection: "column",
              gap: 6,
            }}
          >
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 9,
                letterSpacing: "0.18em",
                textTransform: "uppercase",
                color: "oklch(0.85 0.14 150)",
              }}
            >
              ⚗ pipeline ativo · {pipelineSummary.length} filtro
              {pipelineSummary.length === 1 ? "" : "s"}
            </div>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                color: "var(--vf-text)",
                wordBreak: "break-all",
              }}
            >
              {pipelineSummary.join(" → ")}
            </div>
          </div>
        )}

        {/* Log stream */}
        <div
          ref={logRef}
          style={{
            height: hasFailed ? 120 : 220,
            padding: 16,
            background: "rgba(0,0,0,0.45)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 12,
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            color: "var(--vf-text-dim)",
            overflowY: "auto",
            lineHeight: 1.6,
          }}
        >
          {logs.map((line, i) => (
            <div
              key={i}
              style={{
                color: line.startsWith("$")
                  ? taskAccent
                  : line.startsWith(">")
                    ? "var(--vf-text)"
                    : "var(--vf-text-muted)",
              }}
            >
              {line}
            </div>
          ))}
        </div>

        {/* Action buttons */}
        <div
          style={{
            display: "flex",
            gap: 10,
            marginTop: 18,
            justifyContent: "flex-end",
          }}
        >
          {isRunning && (
            <button
              type="button"
              onClick={onClose}
              style={{
                padding: "10px 18px",
                background: "transparent",
                border: "1px solid var(--vf-panel-stroke)",
                borderRadius: 10,
                color: "var(--vf-text-dim)",
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                cursor: "pointer",
              }}
            >
              Minimizar
            </button>
          )}
          {isFinished && (
            <>
              <button
                type="button"
                onClick={onClose}
                style={{
                  padding: "10px 18px",
                  background: "transparent",
                  border: "1px solid var(--vf-panel-stroke)",
                  borderRadius: 10,
                  color: "var(--vf-text-dim)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  cursor: "pointer",
                }}
              >
                Fechar
              </button>
              {!hasFailed && onViewResults && (
                <button
                  type="button"
                  onClick={onViewResults}
                  style={{
                    padding: "10px 18px",
                    background: "var(--accent-soft)",
                    border: `1px solid ${taskAccent}`,
                    borderRadius: 10,
                    color: "var(--vf-text)",
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                    boxShadow: "inset 0 0 14px var(--accent-glow)",
                    cursor: "pointer",
                  }}
                >
                  ↗ Ver resultados
                </button>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
