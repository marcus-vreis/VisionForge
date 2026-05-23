import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  fetchResult,
  fetchStatus,
  runExperiment,
} from "../api/client";
import type { RunResult, RunStatus, TrainingEvent } from "../types/run";

interface ExperimentState {
  status: RunStatus;
  result: RunResult | null;
  error: string | null;
  validationErrors: ValidationError[];
  progressEvents: TrainingEvent[];
  submit: (config: Record<string, unknown>) => Promise<void>;
  reset: () => void;
}

export interface ValidationError {
  field: string[];
  message: string;
}

const SECTION_LABELS: Record<string, string> = {
  model: "Modelo",
  training: "Treinamento",
  data: "Dataset",
  output: "Saída",
  classification: "Classificação",
  transforms: "Transformações",
  preprocessing: "Pré-processamento",
  steps: "Filtro",
  scheduler: "Scheduler",
  device: "Dispositivo",
};

/** Preprocessing filter ids whose Pydantic errors should appear with a
 * human-friendly name in field path summaries. */
const PREPROCESS_KIND_LABELS: Record<string, string> = {
  gaussian_blur: "Gaussian blur",
  median_blur: "Median blur",
  unsharp: "Unsharp mask",
  edges: "Edges",
  emboss: "Emboss",
  grayscale: "Grayscale",
  equalize: "Equalize",
  autocontrast: "Autocontrast",
  wavelet: "Wavelet",
};

const FIELD_LABELS: Record<string, string> = {
  name: "Nome",
  task: "Tipo de tarefa",
  num_classes: "Nº de classes",
  pretrained: "Pesos pré-treinados",
  weights_path: "Caminho dos pesos",
  learning_rate: "Learning Rate",
  epochs: "Épocas",
  batch_size: "Batch size",
  early_stopping_patience: "Early stop",
  optimizer: "Otimizador",
  weight_decay: "Weight decay",
  seed: "Seed",
  base_dir: "Diretório base",
  train_dir: "Subpasta treino",
  val_dir: "Subpasta validação",
  test_dir: "Subpasta teste",
  num_workers: "Workers",
  pin_memory: "Pin memory",
  image_size: "Tamanho da imagem",
  horizontal_flip: "Flip horizontal",
  rotation_degrees: "Rotação",
  color_jitter: "Color jitter",
  normalize_mean: "Normalização (média)",
  normalize_std: "Normalização (std)",
};

/** Build a user-readable path like "Treinamento › Learning Rate".
 *
 * Numeric segments (Pydantic list index) become "#N" so the user can tell
 * which filter slot in the pipeline failed validation. A known filter kind
 * (passed alongside via the special "kind=foo" pseudo-segment) gets its
 * human label inserted next to the index.
 */
export function humanizeFieldPath(loc: (string | number)[]): string {
  return loc
    .filter((p) => p !== "body")
    .map((p) => {
      if (typeof p === "number") {
        // List index inside steps[] — show as "#N" so the user can tell
        // which filter slot in the pipeline failed validation.
        return `#${p + 1}`;
      }
      const k = String(p);
      const known = PREPROCESS_KIND_LABELS[k];
      if (known) return known;
      return SECTION_LABELS[k] ?? FIELD_LABELS[k] ?? k;
    })
    .filter((s) => s !== "")
    .join(" › ");
}

export function useExperiment(): ExperimentState {
  const [status, setStatus] = useState<RunStatus>({
    status: "idle",
    run_id: null,
    error: null,
  });
  const [result, setResult] = useState<RunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<ValidationError[]>(
    [],
  );
  const [progressEvents, setProgressEvents] = useState<TrainingEvent[]>([]);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const esRef = useRef<EventSource | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const closeEventSource = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }
  }, []);

  const openEventSource = useCallback(() => {
    closeEventSource();
    const es = new EventSource("/api/experiment/events");
    esRef.current = es;

    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as TrainingEvent;
        // Ignore the empty sentinel emitted when no run is active.
        if (!data.event) return;
        setProgressEvents((prev) => [...prev, data]);
        if (data.event === "end") {
          closeEventSource();
        }
      } catch {
        // Malformed frame — skip.
      }
    };

    es.onerror = () => {
      // SSE failed or server closed; fall back to the existing 2s polling.
      closeEventSource();
    };
  }, [closeEventSource]);

  const startPolling = useCallback(
    (_runId: string) => {
      // eslint-disable-line @typescript-eslint/no-unused-vars
      stopPolling();
      pollRef.current = setInterval(async () => {
        try {
          const s = await fetchStatus();
          setStatus(s);

          if (s.status === "completed" && s.run_id) {
            stopPolling();
            try {
              const r = await fetchResult(s.run_id);
              setResult(r);
            } catch (e) {
              const msg =
                e instanceof ApiError
                  ? e.message
                  : "Falha ao buscar resultados do experimento.";
              setError(msg);
              setStatus({ status: "failed", run_id: s.run_id, error: msg });
            }
          } else if (s.status === "failed") {
            stopPolling();
            setError(
              s.error ?? "O experimento falhou sem mensagem detalhada.",
            );
          }
        } catch (e) {
          stopPolling();
          const msg =
            e instanceof ApiError
              ? e.message
              : "Conexão com o servidor perdida durante o polling.";
          setError(msg);
          setStatus((prev) => ({
            status: "failed",
            run_id: prev.run_id,
            error: msg,
          }));
        }
      }, 2000);
    },
    [stopPolling],
  );

  useEffect(() => () => {
    stopPolling();
    closeEventSource();
  }, [stopPolling, closeEventSource]);

  const submit = useCallback(
    async (config: Record<string, unknown>) => {
      setError(null);
      setResult(null);
      setValidationErrors([]);
      setProgressEvents([]);

      try {
        const res = await runExperiment(config);
        setStatus({ status: "running", run_id: res.run_id, error: null });
        openEventSource();
        startPolling(res.run_id);
      } catch (e) {
        if (e instanceof ApiError) {
          if (e.status === 422 && e.validationErrors) {
            setValidationErrors(
              e.validationErrors.map((err) => ({
                field: err.loc
                  .filter((l) => l !== "body")
                  .map((l) => String(l)),
                message: err.msg,
              })),
            );
            setError(
              `${e.validationErrors.length} campo(s) com erro de validação. Confira os destaques no formulário.`,
            );
            return;
          }
          if (e.status === 409) {
            setError("Já existe um experimento em execução. Aguarde terminar.");
            return;
          }
          if (e.status === 0) {
            setError(e.message);
            return;
          }
          setError(e.message);
          return;
        }
        if (e instanceof Error) {
          setError(`Erro inesperado: ${e.message}`);
          return;
        }
        setError("Erro desconhecido ao iniciar o experimento.");
      }
    },
    [startPolling, openEventSource],
  );

  const reset = useCallback(() => {
    stopPolling();
    closeEventSource();
    setStatus({ status: "idle", run_id: null, error: null });
    setResult(null);
    setError(null);
    setValidationErrors([]);
    setProgressEvents([]);
  }, [stopPolling, closeEventSource]);

  return { status, result, error, validationErrors, progressEvents, submit, reset };
}
