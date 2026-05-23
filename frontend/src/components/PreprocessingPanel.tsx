import { useEffect, useState } from "react";
import {
  artifactUrl,
  previewPreprocess,
  type PreprocessPreviewResponse,
} from "../api/client";

export interface PreprocessingStep {
  kind: string;
  params: Record<string, string | number>;
}

interface PreprocessingPanelProps {
  baseDir: string;
  steps: PreprocessingStep[];
  onChange: (steps: PreprocessingStep[]) => void;
}

const KNOWN_KINDS = [
  "gaussian_blur",
  "median_blur",
  "unsharp",
  "edges",
  "emboss",
  "grayscale",
  "equalize",
  "autocontrast",
  "wavelet",
] as const;

const KIND_LABELS: Record<string, string> = {
  gaussian_blur: "Gaussian blur",
  median_blur: "Median blur",
  unsharp: "Unsharp mask",
  edges: "Edges (Sobel)",
  emboss: "Emboss",
  grayscale: "Grayscale",
  equalize: "Equalize (CLAHE)",
  autocontrast: "Autocontrast",
  wavelet: "Wavelet (Haar)",
};

const DEFAULT_PARAMS: Record<string, Record<string, string | number>> = {
  gaussian_blur: { radius: 2.0 },
  median_blur: { size: 3 },
  unsharp: { radius: 2.0, percent: 150, threshold: 3 },
  autocontrast: { cutoff: 1.0 },
  wavelet: { band: "LL" },
};

/** Pipeline builder + live preview for image preprocessing.
 *
 * The pipeline is a controlled value: ``steps`` lives in the parent (typically
 * ``formData.data.preprocessing.steps``) so it round-trips through YAML
 * export/import and gets submitted to /api/experiment/run. The local state
 * here only owns the preview response.
 */
export function PreprocessingPanel({
  baseDir,
  steps,
  onChange,
}: PreprocessingPanelProps) {
  const [preview, setPreview] = useState<PreprocessPreviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset preview when the dataset changes — stale paths would 404.
  useEffect(() => {
    setPreview(null);
    setError(null);
  }, [baseDir]);

  const addStep = (kind: string) => {
    const params = DEFAULT_PARAMS[kind] ?? {};
    onChange([...steps, { kind, params: { ...params } }]);
  };

  const removeStep = (index: number) => {
    onChange(steps.filter((_, i) => i !== index));
  };

  const moveStep = (index: number, dir: -1 | 1) => {
    const target = index + dir;
    if (target < 0 || target >= steps.length) return;
    const next = [...steps];
    [next[index], next[target]] = [next[target], next[index]];
    onChange(next);
  };

  const updateParam = (
    index: number,
    key: string,
    value: string | number,
  ) => {
    onChange(
      steps.map((s, i) =>
        i === index ? { ...s, params: { ...s.params, [key]: value } } : s,
      ),
    );
  };

  const clearSteps = () => {
    onChange([]);
    setPreview(null);
  };

  const runPreview = async () => {
    if (!baseDir.trim()) {
      setError("Defina um diretório base antes de gerar preview.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const payload = steps.map((s) => ({ kind: s.kind, ...s.params }));
      const resp = await previewPreprocess(baseDir, payload);
      if (resp.message) {
        setError(resp.message);
        setPreview(null);
      } else {
        setPreview(resp);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Falha ao gerar preview.");
      setPreview(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        marginTop: 18,
        padding: 18,
        background: "rgba(255,255,255,0.02)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 12,
        display: "flex",
        flexDirection: "column",
        gap: 14,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              letterSpacing: "0.20em",
              textTransform: "uppercase",
              color: "var(--vf-text-muted)",
            }}
          >
            // pré-processamento (filtros)
          </div>
          {steps.length > 0 && (
            <span
              title="Estes filtros serão aplicados durante o treino, antes de augmentation e normalização"
              style={{
                padding: "3px 9px",
                fontFamily: "var(--font-mono)",
                fontSize: 9,
                letterSpacing: "0.14em",
                textTransform: "uppercase",
                color: "oklch(0.88 0.15 150)",
                background: "oklch(0.72 0.16 150 / 0.14)",
                border: "1px solid oklch(0.72 0.16 150 / 0.45)",
                borderRadius: 999,
              }}
            >
              {steps.length} filtro{steps.length === 1 ? "" : "s"} ativo
              {steps.length === 1 ? "" : "s"} no treino
            </span>
          )}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {steps.length > 0 && (
            <button
              type="button"
              onClick={clearSteps}
              title="Remover todos os filtros do pipeline"
              style={{
                padding: "8px 12px",
                background: "transparent",
                border: "1px solid var(--vf-panel-stroke)",
                borderRadius: 8,
                color: "var(--vf-text-dim)",
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                letterSpacing: "0.10em",
                textTransform: "uppercase",
                cursor: "pointer",
              }}
            >
              limpar
            </button>
          )}
          <select
            value=""
            onChange={(e) => {
              if (e.target.value) {
                addStep(e.target.value);
                e.target.value = "";
              }
            }}
            style={{
              padding: "8px 10px",
              background: "rgba(0,0,0,0.35)",
              border: "1px solid var(--vf-panel-stroke)",
              borderRadius: 8,
              color: "var(--vf-text)",
              fontFamily: "var(--font-mono)",
              fontSize: 11,
            }}
          >
            <option value="">+ adicionar filtro</option>
            {KNOWN_KINDS.map((k) => (
              <option key={k} value={k}>
                {KIND_LABELS[k] ?? k}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => void runPreview()}
            disabled={loading || steps.length === 0}
            style={{
              padding: "8px 16px",
              background: "var(--accent-soft)",
              border: "1px solid var(--accent-vf)",
              borderRadius: 8,
              color: "var(--vf-text)",
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              letterSpacing: "0.10em",
              textTransform: "uppercase",
              cursor: loading || steps.length === 0 ? "not-allowed" : "pointer",
              opacity: loading || steps.length === 0 ? 0.5 : 1,
            }}
          >
            {loading ? "Gerando…" : "▶ Ver preview"}
          </button>
        </div>
      </div>

      {steps.length === 0 ? (
        <div
          style={{
            padding: "12px 14px",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--vf-text-muted)",
            border: "1px dashed var(--vf-panel-stroke)",
            borderRadius: 10,
            textAlign: "center",
            lineHeight: 1.6,
          }}
        >
          Pipeline vazio — clique em "+ adicionar filtro".
          <div style={{ fontSize: 10, marginTop: 4, opacity: 0.7 }}>
            O pipeline configurado aqui roda <strong>antes</strong> de augmentation
            e normalização, em todos os splits (treino / val / teste).
          </div>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {steps.map((step, i) => (
            <StepRow
              key={`${step.kind}-${i}`}
              step={step}
              index={i}
              total={steps.length}
              onRemove={() => removeStep(i)}
              onMoveUp={() => moveStep(i, -1)}
              onMoveDown={() => moveStep(i, 1)}
              onParamChange={(k, v) => updateParam(i, k, v)}
            />
          ))}
        </div>
      )}

      {error && (
        <div
          style={{
            padding: "10px 14px",
            background: "oklch(0.704 0.191 22.216 / 0.10)",
            border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
            borderRadius: 10,
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "oklch(0.85 0.14 22)",
          }}
        >
          {error}
        </div>
      )}

      {preview && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
            gap: 10,
          }}
        >
          <PreviewTile label="Original" artifact={preview.original} />
          {preview.steps.map((s, i) => (
            <PreviewTile
              key={`${s.kind}-${i}`}
              label={`${i + 1}. ${KIND_LABELS[s.kind] ?? s.kind}`}
              artifact={s.artifact}
            />
          ))}
          {preview.steps.length > 0 && preview.final !== preview.steps[preview.steps.length - 1].artifact && (
            <PreviewTile label="Final" artifact={preview.final} />
          )}
        </div>
      )}
    </div>
  );
}

interface StepRowProps {
  step: PreprocessingStep;
  index: number;
  total: number;
  onRemove: () => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
  onParamChange: (key: string, value: string | number) => void;
}

function StepRow({
  step,
  index,
  total,
  onRemove,
  onMoveUp,
  onMoveDown,
  onParamChange,
}: StepRowProps) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "8px 12px",
        background: "rgba(0,0,0,0.30)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 8,
      }}
    >
      <span
        style={{
          minWidth: 22,
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          color: "var(--vf-text-muted)",
        }}
      >
        {index + 1}.
      </span>
      <span
        style={{
          minWidth: 130,
          fontFamily: "var(--font-mono)",
          fontSize: 12,
          color: "var(--vf-text)",
        }}
      >
        {KIND_LABELS[step.kind] ?? step.kind}
      </span>
      <div style={{ display: "flex", gap: 8, flex: 1, flexWrap: "wrap" }}>
        {Object.entries(step.params).map(([k, v]) => (
          <label
            key={k}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "var(--vf-text-dim)",
            }}
          >
            <span>{k}:</span>
            <input
              type={typeof v === "number" ? "number" : "text"}
              value={String(v)}
              step={typeof v === "number" && Number.isInteger(v) ? 1 : "any"}
              onChange={(e) => {
                const raw = e.target.value;
                if (typeof v === "number") {
                  const parsed = parseFloat(raw);
                  if (!Number.isNaN(parsed)) onParamChange(k, parsed);
                } else {
                  onParamChange(k, raw);
                }
              }}
              style={{
                width: 70,
                padding: "3px 6px",
                background: "rgba(0,0,0,0.40)",
                border: "1px solid var(--vf-panel-stroke)",
                borderRadius: 4,
                color: "var(--vf-text)",
                fontFamily: "var(--font-mono)",
                fontSize: 11,
              }}
            />
          </label>
        ))}
      </div>
      <div style={{ display: "flex", gap: 4 }}>
        <button
          type="button"
          onClick={onMoveUp}
          disabled={index === 0}
          title="Mover para cima"
          style={btnIconStyle(index === 0)}
        >
          ↑
        </button>
        <button
          type="button"
          onClick={onMoveDown}
          disabled={index === total - 1}
          title="Mover para baixo"
          style={btnIconStyle(index === total - 1)}
        >
          ↓
        </button>
        <button
          type="button"
          onClick={onRemove}
          title="Remover"
          style={btnIconStyle(false)}
        >
          ×
        </button>
      </div>
    </div>
  );
}

function btnIconStyle(disabled: boolean): React.CSSProperties {
  return {
    width: 24,
    height: 24,
    padding: 0,
    background: "transparent",
    border: "1px solid var(--vf-panel-stroke)",
    borderRadius: 4,
    color: "var(--vf-text-dim)",
    fontFamily: "var(--font-mono)",
    fontSize: 12,
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.35 : 1,
  };
}

function PreviewTile({ label, artifact }: { label: string; artifact: string }) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 4,
        background: "rgba(0,0,0,0.30)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 8,
        padding: 6,
      }}
    >
      <img
        src={artifactUrl(artifact)}
        alt={label}
        style={{
          width: "100%",
          height: 140,
          objectFit: "cover",
          borderRadius: 6,
        }}
      />
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--vf-text-dim)",
          textAlign: "center",
        }}
      >
        {label}
      </div>
    </div>
  );
}
