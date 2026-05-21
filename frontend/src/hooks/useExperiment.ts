import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  fetchResult,
  fetchStatus,
  runExperiment,
} from "../api/client";
import type { RunResult, RunStatus } from "../types/run";

interface ExperimentState {
  status: RunStatus;
  result: RunResult | null;
  error: string | null;
  validationErrors: ValidationError[];
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

/** Build a user-readable path like "Treinamento › Learning Rate". */
export function humanizeFieldPath(loc: (string | number)[]): string {
  return loc
    .filter((p) => p !== "body")
    .map((p) => {
      const k = String(p);
      return SECTION_LABELS[k] ?? FIELD_LABELS[k] ?? k;
    })
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
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

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

  useEffect(() => stopPolling, [stopPolling]);

  const submit = useCallback(
    async (config: Record<string, unknown>) => {
      setError(null);
      setResult(null);
      setValidationErrors([]);

      try {
        const res = await runExperiment(config);
        setStatus({ status: "running", run_id: res.run_id, error: null });
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
    [startPolling],
  );

  const reset = useCallback(() => {
    stopPolling();
    setStatus({ status: "idle", run_id: null, error: null });
    setResult(null);
    setError(null);
    setValidationErrors([]);
  }, [stopPolling]);

  return { status, result, error, validationErrors, submit, reset };
}
