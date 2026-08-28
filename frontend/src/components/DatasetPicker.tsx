import { useEffect, useState } from "react";
import {
  ApiError,
  detectDatasetSplits,
  pickDatasetFolder,
} from "../api/client";
import { SelectField, type SelectOption } from "./controls/SelectField";
import { paramHelp } from "../lib/param-help";
import { TextField } from "./controls/TextField";

interface DatasetPickerProps {
  baseDir: string;
  trainDir: string;
  valDir: string;
  testDir: string;
  onChange: (next: {
    base_dir?: string;
    train_dir?: string;
    val_dir?: string;
    test_dir?: string;
  }) => void;
}

type FeedbackKind = "info" | "success" | "warning" | "error";

interface Feedback {
  kind: FeedbackKind;
  message: string;
}

const FEEDBACK_COLORS: Record<FeedbackKind, { bg: string; border: string; fg: string }> = {
  info: {
    bg: "rgba(255,255,255,0.04)",
    border: "var(--vf-panel-stroke)",
    fg: "var(--vf-text-dim)",
  },
  success: {
    bg: "oklch(0.72 0.16 150 / 0.10)",
    border: "oklch(0.72 0.16 150 / 0.45)",
    fg: "oklch(0.85 0.16 150)",
  },
  warning: {
    bg: "oklch(0.83 0.16 80 / 0.10)",
    border: "oklch(0.83 0.16 80 / 0.45)",
    fg: "oklch(0.88 0.16 80)",
  },
  error: {
    bg: "oklch(0.704 0.191 22.216 / 0.10)",
    border: "oklch(0.704 0.191 22.216 / 0.45)",
    fg: "oklch(0.85 0.14 22)",
  },
};

const buttonStyle: React.CSSProperties = {
  padding: "10px 16px",
  background: "var(--accent-soft)",
  border: "1px solid var(--accent-vf)",
  borderRadius: 10,
  color: "var(--vf-text)",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  letterSpacing: "0.10em",
  textTransform: "uppercase",
  cursor: "pointer",
  whiteSpace: "nowrap",
};

// Browser File System Access API only exposes folder *names*, never absolute
// paths. To deliver a real absolute path we hand off to /api/dataset/pick,
// which opens a native (tkinter) dialog on the server — safe because the GUI
// is always local.

/** Dataset folder input with auto-detect + manual override selectors. */
export function DatasetPicker({
  baseDir,
  trainDir,
  valDir,
  testDir,
  onChange,
}: DatasetPickerProps) {
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [candidates, setCandidates] = useState<string[]>([]);

  // Auto-detecta os splits sempre que base_dir muda — substitui o antigo botão
  // "Detectar splits". Debounce evita chamar a API a cada tecla digitada; o
  // seletor nativo também muda base_dir, então a detecção dispara após ele.
  // Todo setState roda dentro do timer (assíncrono) para não disparar render
  // em cascata a partir do corpo do effect.
  useEffect(() => {
    let alive = true;
    const trimmed = baseDir.trim();
    const timer = setTimeout(
      () => {
        if (!trimmed) {
          setCandidates([]);
          setFeedback(null);
          return;
        }
        void (async () => {
          setFeedback({ kind: "info", message: "Analisando subpastas…" });
          try {
            const result = await detectDatasetSplits(trimmed);
            if (!alive) return;
            setCandidates(result.candidates);
            const next: {
              train_dir?: string;
              val_dir?: string;
              test_dir?: string;
            } = {};
            if (result.train_dir) next.train_dir = result.train_dir;
            if (result.val_dir) next.val_dir = result.val_dir;
            if (result.test_dir) next.test_dir = result.test_dir;
            if (Object.keys(next).length > 0) onChange(next);
            if (result.detected) {
              setFeedback({ kind: "success", message: result.message });
            } else if (result.candidates.length > 0) {
              setFeedback({ kind: "warning", message: result.message });
            } else {
              setFeedback({ kind: "error", message: result.message });
            }
          } catch (e) {
            if (!alive) return;
            const msg =
              e instanceof ApiError
                ? e.message
                : e instanceof Error
                  ? e.message
                  : "Falha ao detectar splits do dataset.";
            setFeedback({ kind: "error", message: msg });
          }
        })();
      },
      trimmed ? 400 : 0,
    );
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [baseDir]); // eslint-disable-line react-hooks/exhaustive-deps

  const handlePickFolder = async () => {
    setFeedback({ kind: "info", message: "Abrindo seletor nativo do sistema…" });
    try {
      const res = await pickDatasetFolder();
      if (res.cancelled) {
        setFeedback({
          kind: "info",
          message: res.message ?? "Seleção cancelada.",
        });
        return;
      }
      onChange({ base_dir: res.path });
      setFeedback({
        kind: "success",
        message: `Pasta selecionada: ${res.path}`,
      });
    } catch (e) {
      const msg =
        e instanceof ApiError
          ? e.message
          : e instanceof Error
            ? e.message
            : "Falha ao abrir o seletor de pastas.";
      setFeedback({ kind: "error", message: msg });
    }
  };

  const splitOptions: SelectOption[] = candidates.map((c) => ({
    value: c,
    label: c,
  }));

  const splitField = (
    label: string,
    value: string,
    onChangeKey: "train_dir" | "val_dir" | "test_dir",
  ) => {
    const help = paramHelp(onChangeKey);
    if (candidates.length > 0) {
      const opts: SelectOption[] = value && !candidates.includes(value)
        ? [{ value, label: `${value} (manual)` }, ...splitOptions]
        : splitOptions;
      return (
        <SelectField
          label={label}
          value={value || ""}
          onChange={(v) => onChange({ [onChangeKey]: v })}
          options={opts}
          help={help}
        />
      );
    }
    return (
      <TextField
        label={label}
        value={value}
        onChange={(v) => onChange({ [onChangeKey]: v })}
        mono
        help={help}
      />
    );
  };

  const fb = feedback ? FEEDBACK_COLORS[feedback.kind] : null;

  return (
    <div>
      <div
        data-tour="dataset"
        style={{
          padding: 18,
          background: "rgba(255,255,255,0.02)",
          border: "1px dashed var(--vf-panel-stroke)",
          borderRadius: 12,
          display: "flex",
          flexDirection: "column",
          gap: 14,
          marginBottom: 14,
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-end", gap: 12 }}>
          <div style={{ flex: 1 }}>
            <TextField
              label="Diretório base do dataset"
              value={baseDir}
              onChange={(v) => onChange({ base_dir: v })}
              placeholder="ex: C:/datasets/coffee  ou  /home/user/data"
              mono
              hint="Pasta raiz que contém treino, validação e teste."
              help={paramHelp("base_dir")}
            />
          </div>
          <button
            type="button"
            onClick={() => void handlePickFolder()}
            style={{
              ...buttonStyle,
              background: "transparent",
              border: "1px solid var(--vf-panel-stroke)",
              color: "var(--vf-text-dim)",
            }}
            title="Abrir seletor nativo do sistema (retorna o caminho absoluto)"
          >
            📁 Escolher pasta
          </button>
        </div>

        {feedback && fb && (
          <div
            style={{
              padding: "10px 14px",
              background: fb.bg,
              border: `1px solid ${fb.border}`,
              borderRadius: 10,
              fontFamily: "var(--font-mono)",
              fontSize: 12,
              color: fb.fg,
              lineHeight: 1.5,
            }}
          >
            {feedback.message}
          </div>
        )}
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 18,
          marginBottom: 16,
        }}
      >
        {splitField("Subpasta treino", trainDir, "train_dir")}
        {splitField("Subpasta validação", valDir, "val_dir")}
        {splitField("Subpasta teste", testDir, "test_dir")}
      </div>
    </div>
  );
}
