import { useState } from "react";
import {
  ApiError,
  detectDatasetSplits,
  type DatasetDetectResponse,
} from "../api/client";
import { SelectField, type SelectOption } from "./controls/SelectField";
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

/** Try the File System Access API (Chromium) to pick a folder. */
async function tryShowDirectoryPicker(): Promise<string | null> {
  const w = window as unknown as {
    showDirectoryPicker?: () => Promise<{ name: string }>;
  };
  if (typeof w.showDirectoryPicker !== "function") return null;
  try {
    const handle = await w.showDirectoryPicker();
    // Browsers do not expose absolute paths. Return the folder name so the
    // user can prepend its parent path if needed.
    return handle.name;
  } catch {
    return null;
  }
}

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
  const [detecting, setDetecting] = useState(false);

  const handleDetect = async () => {
    if (!baseDir.trim()) {
      setFeedback({
        kind: "error",
        message:
          "Informe primeiro o caminho do diretório base do dataset.",
      });
      return;
    }
    setDetecting(true);
    setFeedback({ kind: "info", message: "Analisando subpastas…" });
    try {
      const result: DatasetDetectResponse = await detectDatasetSplits(baseDir);
      setCandidates(result.candidates);
      const next: Partial<DatasetPickerProps> = {};
      if (result.train_dir) next.trainDir = result.train_dir;
      if (result.val_dir) next.valDir = result.val_dir;
      if (result.test_dir) next.testDir = result.test_dir;
      onChange({
        base_dir: result.base_dir,
        train_dir: next.trainDir,
        val_dir: next.valDir,
        test_dir: next.testDir,
      });

      if (result.detected) {
        setFeedback({ kind: "success", message: result.message });
      } else if (result.candidates.length > 0) {
        setFeedback({ kind: "warning", message: result.message });
      } else {
        setFeedback({ kind: "error", message: result.message });
      }
    } catch (e) {
      const msg =
        e instanceof ApiError
          ? e.message
          : e instanceof Error
            ? e.message
            : "Falha ao detectar splits do dataset.";
      setFeedback({ kind: "error", message: msg });
    } finally {
      setDetecting(false);
    }
  };

  const handlePickFolder = async () => {
    const name = await tryShowDirectoryPicker();
    if (name) {
      // Browser sandbox: we only know the folder *name*. Suggest the user
      // append it to a parent path. If baseDir is empty, set it to name as
      // a hint; otherwise leave untouched.
      if (!baseDir) onChange({ base_dir: name });
      setFeedback({
        kind: "info",
        message:
          `Pasta selecionada: '${name}'. ` +
          "Edite o caminho acima para incluir o caminho absoluto se necessário.",
      });
    } else {
      setFeedback({
        kind: "info",
        message:
          "Seu navegador não suporta seleção nativa de pasta. " +
          "Cole o caminho absoluto no campo acima.",
      });
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
        />
      );
    }
    return (
      <TextField
        label={label}
        value={value}
        onChange={(v) => onChange({ [onChangeKey]: v })}
        mono
      />
    );
  };

  const fb = feedback ? FEEDBACK_COLORS[feedback.kind] : null;

  return (
    <div>
      <div
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
            title="Abrir seletor nativo do navegador (Chrome/Edge)"
          >
            📁 Escolher
          </button>
          <button
            type="button"
            onClick={() => void handleDetect()}
            disabled={detecting}
            style={{
              ...buttonStyle,
              opacity: detecting ? 0.5 : 1,
              cursor: detecting ? "wait" : "pointer",
            }}
          >
            {detecting ? "Detectando…" : "↻ Detectar splits"}
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
