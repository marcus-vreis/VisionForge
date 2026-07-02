import { createContext, useContext, useEffect, useRef, useState } from "react";
import { fetchSystemInfo, pickCheckpointFile } from "../api/client";
import { humanizeFieldPath, type ValidationError } from "../hooks/useExperiment";
import type { JsonSchema } from "../types/schema";
import type { TaskDefinition } from "../types/tasks";
import {
  exportConfigToYaml,
  importConfigFromYaml,
  validateParsedConfig,
} from "../lib/yaml-config";
import { AugmentPreview } from "./AugmentPreview";
import { DatasetPicker } from "./DatasetPicker";
import { DatasetStats } from "./DatasetStats";
import {
  PreprocessingPanel,
  type PreprocessingStep as PreprocessingUIStep,
} from "./PreprocessingPanel";
import { resolveKind } from "./field-renderer";
import {
  coerceGridValue,
  isGridableField,
  suggestNextGridValue,
  validateGridValue,
} from "../lib/grid-axis";
import {
  NumberField,
  SelectField,
  Segmented,
  TextField,
  Toggle,
} from "./controls";

interface ParamPanelProps {
  task: TaskDefinition;
  schema: JsonSchema | null;
  formData: Record<string, unknown>;
  setFormData: React.Dispatch<React.SetStateAction<Record<string, unknown>>>;
  validationErrors: ValidationError[];
}

/** Section labels for config sub-models. */
const SECTION_LABELS: Record<string, string> = {
  model: "Modelo",
  training: "Treinamento",
  data: "Dataset",
  output: "Saída",
  classification: "Classificação",
  transforms: "Transformações",
};

/** Human-readable field labels. Dot-path keys override leaf-name keys. */
const FIELD_LABELS: Record<string, string> = {
  "model.name": "Arquitetura",
  name: "Nome do experimento",
  task: "Tipo de tarefa",
  block: "Bloco",
  num_classes: "Nº de classes",
  pretrained: "Pesos pré-treinados",
  weights_path: "Caminho dos pesos",
  learning_rate: "Learning Rate",
  epochs: "Épocas",
  batch_size: "Batch size",
  early_stopping_patience: "Early stop (paciência)",
  optimizer: "Otimizador",
  weight_decay: "Weight decay",
  seed: "Seed",
  deterministic: "Determinístico (lento)",
  mixed_precision: "Precisão mista (AMP)",
  kind: "Tipo",
  step_size: "Step size",
  gamma: "Gamma",
  patience: "Paciência",
  factor: "Fator",
  min_lr: "LR mínimo",
  base_dir: "Diretório base",
  train_dir: "Subdir treino",
  val_dir: "Subdir validação",
  test_dir: "Subdir teste",
  num_workers: "Workers",
  pin_memory: "Pin memory",
  image_size: "Tamanho da imagem",
  horizontal_flip: "Flip horizontal",
  rotation_degrees: "Rotação (graus)",
  color_jitter: "Color jitter",
  normalize_mean: "Normalização (média)",
  normalize_std: "Normalização (std)",
  // Cross-validation
  n_folds: "Nº de folds",
  stratified: "Stratified",
  shuffle: "Shuffle",
  fold_seed: "Fold seed",
  // Transfer learning
  mode: "Modo",
  unfreeze_from_layer: "Descongelar a partir de",
  backbone_lr_multiplier: "LR do backbone (×)",
  // Model comparison
  model_names: "Arquiteturas",
  metric: "Métrica de ranking",
};

function resolveSchema(
  schema: JsonSchema,
  defs: Record<string, JsonSchema>,
): JsonSchema {
  if (schema.$ref) {
    const refName = schema.$ref.split("/").pop()!;
    return defs[refName] ?? schema;
  }
  return schema;
}

/** Convert the schema-flat preprocessing step (``{kind, ...params}``) the
 * backend expects into the nested ``{kind, params}`` shape the UI panel uses.
 * The backend's PreprocessingStep model uses extra="allow", so any field other
 * than ``kind`` is treated as a filter parameter. */
function toUIPreprocessingSteps(
  raw: unknown,
): PreprocessingUIStep[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((s): s is Record<string, unknown> => typeof s === "object" && s !== null)
    .map((s) => {
      const { kind, ...rest } = s as { kind?: unknown } & Record<string, unknown>;
      const params: Record<string, string | number> = {};
      for (const [k, v] of Object.entries(rest)) {
        if (typeof v === "number" || typeof v === "string") {
          params[k] = v;
        }
      }
      return { kind: String(kind ?? ""), params };
    })
    .filter((s) => s.kind !== "");
}

/** Inverse of toUIPreprocessingSteps — flattens params back into the
 * schema-level shape so the Pydantic ``PreprocessingStep`` model accepts it. */
function fromUIPreprocessingSteps(
  steps: PreprocessingUIStep[],
): Array<Record<string, unknown>> {
  return steps.map((s) => ({ kind: s.kind, ...s.params }));
}


interface SchedulerFieldsProps {
  schema: JsonSchema;
  defs: Record<string, JsonSchema>;
  value: Record<string, unknown>;
  onChange: (v: Record<string, unknown>) => void;
  errors: ValidationError[];
}

/** Scheduler config UI — picks ``kind`` and shows only the relevant params. */
function SchedulerFields({
  schema,
  defs,
  value,
  onChange,
  errors,
}: SchedulerFieldsProps) {
  const resolved = resolveSchema(schema, defs);
  if (!resolved.properties) return null;

  const kind = (value["kind"] as string) ?? "none";
  const setParam = (k: string, v: unknown) => onChange({ ...value, [k]: v });

  // Only show params relevant to the selected kind so the form stays tidy.
  const visibleParams: Record<string, string[]> = {
    none: [],
    cosine: [],
    step: ["step_size", "gamma"],
    plateau: ["patience", "factor", "min_lr"],
  };
  const shown = visibleParams[kind] ?? [];

  return (
    <div
      style={{
        marginTop: 4,
        marginBottom: 26,
        padding: 14,
        background: "rgba(255,255,255,0.015)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 10,
        display: "flex",
        flexDirection: "column",
        gap: 14,
      }}
    >
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
        }}
      >
        // learning-rate scheduler
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr 1fr",
          gap: 14,
        }}
      >
        {resolved.properties["kind"] && (
          <SchemaFieldVF
            name="kind"
            schema={resolved.properties["kind"]}
            defs={defs}
            value={kind}
            onChange={(v) => setParam("kind", v)}
            errors={errors}
            path={["training", "scheduler", "kind"]}
          />
        )}
        {shown.map((paramKey) =>
          resolved.properties?.[paramKey] ? (
            <SchemaFieldVF
              key={paramKey}
              name={paramKey}
              schema={resolved.properties[paramKey]}
              defs={defs}
              value={value[paramKey]}
              onChange={(v) => setParam(paramKey, v)}
              errors={errors}
              path={["training", "scheduler", paramKey]}
            />
          ) : null,
        )}
      </div>
    </div>
  );
}


/** NumberField for num_workers with an "auto" toggle backed by /api/system/info. */
function NumWorkersField({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  const [auto, setAuto] = useState(false);
  const [suggested, setSuggested] = useState<number | null>(null);

  useEffect(() => {
    let alive = true;
    fetchSystemInfo()
      .then((info) => alive && setSuggested(info.suggested_workers))
      .catch(() => {
        /* CPU probe is non-critical — leave default value */
      });
    return () => {
      alive = false;
    };
  }, []);

  const handleToggleAuto = () => {
    if (auto) {
      setAuto(false);
      return;
    }
    if (suggested !== null) {
      onChange(suggested);
    }
    setAuto(true);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 9,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 8,
        }}
      >
        <span>Workers</span>
        <button
          type="button"
          onClick={handleToggleAuto}
          disabled={suggested === null}
          style={{
            padding: "2px 8px",
            background: auto ? "var(--accent-soft)" : "rgba(255,255,255,0.04)",
            border: `1px solid ${auto ? "var(--accent-vf)" : "var(--vf-panel-stroke)"}`,
            borderRadius: 6,
            color: auto ? "var(--vf-text)" : "var(--vf-text-dim)",
            fontFamily: "var(--font-mono)",
            fontSize: 9,
            letterSpacing: "0.10em",
            textTransform: "uppercase",
            cursor: suggested === null ? "not-allowed" : "pointer",
            opacity: suggested === null ? 0.5 : 1,
          }}
          title={
            suggested === null
              ? "Backend não disponível"
              : `min(cpu_count, 8) = ${suggested}`
          }
        >
          auto
        </button>
      </span>
      <input
        type="number"
        value={value}
        min={0}
        step={1}
        disabled={auto}
        onChange={(e) => {
          const v = parseInt(e.target.value, 10);
          if (!Number.isNaN(v)) onChange(v);
        }}
        style={{
          padding: "8px 12px",
          background: auto ? "rgba(0,0,0,0.35)" : "rgba(0,0,0,0.30)",
          border: `1px ${auto ? "dashed" : "solid"} var(--vf-panel-stroke)`,
          borderRadius: 8,
          fontFamily: "var(--font-mono)",
          fontSize: 13,
          color: "var(--vf-text)",
          opacity: auto ? 0.7 : 1,
          cursor: auto ? "not-allowed" : "text",
        }}
      />
    </div>
  );
}


/** weights_path field with a server-side file picker button.
 *
 * When set, the backend bypasses ImageNet weights and loads this .pth/.pt
 * checkpoint into the architecture before training begins. Useful when the
 * researcher wants to fine-tune from a previously trained run of their own.
 */
function WeightsPathField({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  const [picking, setPicking] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const handlePick = async () => {
    setPicking(true);
    setMessage(null);
    try {
      const res = await pickCheckpointFile();
      if (res.cancelled) {
        setMessage(res.message ?? "Cancelado.");
        return;
      }
      onChange(res.path);
    } catch (e) {
      setMessage(e instanceof Error ? e.message : "Falha ao abrir o seletor.");
    } finally {
      setPicking(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 9,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 8,
        }}
      >
        <span>Checkpoint custom (.pth)</span>
        {value && (
          <button
            type="button"
            onClick={() => onChange("")}
            title="Remover checkpoint custom (volta a usar pesos pretrained / random)"
            style={{
              padding: "2px 8px",
              background: "transparent",
              border: "1px solid var(--vf-panel-stroke)",
              borderRadius: 6,
              color: "var(--vf-text-dim)",
              fontFamily: "var(--font-mono)",
              fontSize: 9,
              letterSpacing: "0.10em",
              textTransform: "uppercase",
              cursor: "pointer",
            }}
          >
            limpar
          </button>
        )}
      </span>
      <div style={{ display: "flex", gap: 6, alignItems: "stretch" }}>
        <input
          type="text"
          value={value}
          placeholder="opcional — sobrescreve ImageNet"
          onChange={(e) => onChange(e.target.value)}
          style={{
            flex: 1,
            padding: "8px 12px",
            background: "rgba(0,0,0,0.30)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 8,
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            color: "var(--vf-text)",
          }}
        />
        <button
          type="button"
          onClick={() => void handlePick()}
          disabled={picking}
          style={{
            padding: "0 14px",
            background: "transparent",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 8,
            color: "var(--vf-text-dim)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            cursor: picking ? "wait" : "pointer",
            whiteSpace: "nowrap",
          }}
        >
          📁 {picking ? "…" : "Escolher"}
        </button>
      </div>
      {message && (
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            color: "var(--vf-text-muted)",
          }}
        >
          {message}
        </span>
      )}
    </div>
  );
}


/** Block selector — toggles between simple training and K-Fold cross-validation.
 *
 * These are the two block kinds wired into the GUI today (the backend has a
 * few more: grid_search, transfer_learning, etc. — exposed in future phases).
 * Selecting K-Fold reveals the CrossValidationFields below.
 */
function BlockSelector({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: string) => void;
}) {
  const options = [
    { value: "classification", label: "Treino simples" },
    { value: "cross_validation", label: "K-Fold (CV)" },
    { value: "transfer_learning", label: "Transfer learning" },
    { value: "model_comparison", label: "Comparar modelos" },
    { value: "grid_search", label: "Grid search" },
    { value: "random_search", label: "Random search" },
  ];
  return (
    <div
      style={{
        marginBottom: 22,
        padding: 16,
        background: "rgba(255,255,255,0.02)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 12,
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.20em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
        }}
      >
        // estratégia de experimento
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        {options.map((o) => {
          const selected = value === o.value;
          return (
            <button
              key={o.value}
              type="button"
              onClick={() => onChange(o.value)}
              style={{
                flex: 1,
                padding: "10px 14px",
                background: selected
                  ? "var(--accent-soft)"
                  : "rgba(0,0,0,0.30)",
                border: `1px solid ${selected ? "var(--accent-vf)" : "var(--vf-panel-stroke)"}`,
                borderRadius: 10,
                color: selected ? "var(--vf-text)" : "var(--vf-text-dim)",
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                letterSpacing: "0.08em",
                cursor: "pointer",
                boxShadow: selected ? "inset 0 0 12px var(--accent-glow)" : "none",
                transition: "background 160ms ease, color 160ms ease",
              }}
            >
              {o.label}
            </button>
          );
        })}
      </div>
      {value === "cross_validation" && (
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            color: "var(--vf-text-muted)",
            lineHeight: 1.5,
            marginTop: 2,
          }}
        >
          Treina N modelos em N folds da pasta de treino. Validação por fold;
          normalize_mean/std são recalculados por fold para evitar data leakage.
          Não usa o split de teste — agregação em <code>cv_summary.json</code>.
        </div>
      )}
      {value === "transfer_learning" && (
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            color: "var(--vf-text-muted)",
            lineHeight: 1.5,
            marginTop: 2,
          }}
        >
          Feature extraction (só treina o head) ou fine-tuning (head + backbone
          parcial com LR menor). Útil para fine-tunar em datasets pequenos sem
          destruir as features pré-treinadas.
        </div>
      )}
      {value === "model_comparison" && (
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            color: "var(--vf-text-muted)",
            lineHeight: 1.5,
            marginTop: 2,
          }}
        >
          Treina cada arquitetura selecionada com os mesmos hyperparams e mostra
          o ranking pela métrica escolhida. O campo "Arquitetura" acima é
          ignorado — quem manda é a lista abaixo.
        </div>
      )}
      {value === "grid_search" && (
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            color: "var(--vf-text-muted)",
            lineHeight: 1.5,
            marginTop: 2,
          }}
        >
          Treina <strong>uma vez por combinação</strong> do produto cartesiano
          do espaço definido abaixo. Cada chave é um dot-path (ex:{" "}
          <code>training.learning_rate</code>); o valor é uma lista. Cuidado:
          3×3×2 já são 18 treinos.
        </div>
      )}
      {value === "random_search" && (
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            color: "var(--vf-text-muted)",
            lineHeight: 1.5,
            marginTop: 2,
          }}
        >
          Amostra <code>n_trials</code> configurações independentes do espaço
          abaixo. Cada parâmetro tem um tipo: <code>uniform</code>,{" "}
          <code>log_uniform</code> (LR e weight_decay) ou <code>choice</code>{" "}
          (listas discretas).
        </div>
      )}
    </div>
  );
}


/** Cross-validation hyperparameters — n_folds, stratified, shuffle, fold_seed. */
function CrossValidationFields({
  schema,
  defs,
  value,
  onChange,
  errors,
}: {
  schema: JsonSchema;
  defs: Record<string, JsonSchema>;
  value: Record<string, unknown>;
  onChange: (v: Record<string, unknown>) => void;
  errors: ValidationError[];
}) {
  const resolved = resolveSchema(schema, defs);
  if (!resolved.properties) return null;

  const setParam = (k: string, v: unknown) => onChange({ ...value, [k]: v });
  const fields = ["n_folds", "stratified", "shuffle", "fold_seed"];

  return (
    <div
      style={{
        marginBottom: 26,
        padding: 18,
        background: "rgba(255,255,255,0.02)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 12,
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.20em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
        }}
      >
        // k-fold cross-validation
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: 14,
        }}
      >
        {fields.map((key) =>
          resolved.properties?.[key] ? (
            <SchemaFieldVF
              key={key}
              name={key}
              schema={resolved.properties[key]}
              defs={defs}
              value={value[key]}
              onChange={(v) => setParam(key, v)}
              errors={errors}
              path={["cross_validation", key]}
            />
          ) : null,
        )}
      </div>
    </div>
  );
}


/** Parse a CSV-style list of values into a list with smart type coercion.
 *
 * Numeric tokens become numbers, "true"/"false" become booleans, everything
 * else stays as a string. Empty entries are dropped. Used by GridSearch and
 * RandomSearch (when wired) to keep the editor a single textarea per key.
 */
function parseGridValuesCsv(raw: string): Array<number | boolean | string> {
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s !== "")
    .map((s) => {
      const n = Number(s);
      if (s !== "" && !Number.isNaN(n)) return n;
      if (s.toLowerCase() === "true") return true;
      if (s.toLowerCase() === "false") return false;
      return s;
    });
}

// ── Grid search: per-hyperparameter axis editing ────────────────────────────
// Pure helpers (validation, coercion, gridability) live in lib/grid-axis.ts.

interface GridContextValue {
  enabled: boolean;
  hyperparameters: Record<string, unknown[]>;
  setAxis: (dotPath: string, values: unknown[] | null) => void;
}

const GridContext = createContext<GridContextValue>({
  enabled: false,
  hyperparameters: {},
  setAxis: () => {},
});

const gridAddBtnStyle: React.CSSProperties = {
  marginTop: 8,
  padding: "5px 10px",
  background: "var(--accent-soft)",
  border: "1px solid var(--accent-vf)",
  borderRadius: 7,
  color: "var(--vf-text)",
  fontFamily: "var(--font-mono)",
  fontSize: 10.5,
  letterSpacing: "0.08em",
  textTransform: "uppercase",
  cursor: "pointer",
};

const gridRowControlStyle: React.CSSProperties = {
  flex: 1,
  padding: "8px 10px",
  background: "rgba(0,0,0,0.30)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 8,
  fontFamily: "var(--font-mono)",
  fontSize: 12,
  color: "var(--vf-text)",
  outline: "none",
};

const gridRemoveBtnStyle: React.CSSProperties = {
  width: 30,
  height: 30,
  flexShrink: 0,
  padding: 0,
  background: "transparent",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 8,
  color: "var(--vf-text-dim)",
  fontFamily: "var(--font-mono)",
  fontSize: 14,
  cursor: "pointer",
};

const gridErrStyle: React.CSSProperties = {
  marginTop: 3,
  fontFamily: "var(--font-mono)",
  fontSize: 10.5,
  color: "oklch(0.704 0.191 22.216)",
};

const gridAxisTagStyle: React.CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  letterSpacing: "0.18em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
};

const gridBannerStyle: React.CSSProperties = {
  marginBottom: 22,
  padding: "14px 16px",
  background: "var(--accent-soft)",
  border: "1px solid var(--accent-vf)",
  borderRadius: 12,
  color: "var(--vf-text)",
};

/** Inline editor for the extra grid values of one hyperparameter.
 *
 * The field's normal control owns value #1 of the axis; this renders the "+"
 * button and values #2, #3… so the axis is ``[primary, ...extras]``. Collapsing
 * back to a single value removes the field from the search space. */
function GridAxisExtension({
  dotPath,
  name,
  schema,
  primaryValue,
}: {
  dotPath: string;
  name: string;
  schema: JsonSchema;
  primaryValue: unknown;
}) {
  const ctx = useContext(GridContext);
  const axis = ctx.hyperparameters[dotPath];
  const values = Array.isArray(axis) ? axis : [];
  const extras = values.slice(1);
  const enumOpts = schema.enum?.map(String);

  const addValue = () => {
    const base = values.length ? [...values] : [primaryValue];
    base.push(suggestNextGridValue(base[base.length - 1], name, schema));
    ctx.setAxis(dotPath, base);
  };
  const setExtra = (i: number, raw: string) => {
    const next = [...values];
    next[i + 1] = coerceGridValue(raw, Boolean(enumOpts));
    ctx.setAxis(dotPath, next);
  };
  const removeExtra = (i: number) => {
    const next = values.filter((_, idx) => idx !== i + 1);
    ctx.setAxis(dotPath, next.length <= 1 ? null : next);
  };

  if (extras.length === 0) {
    return (
      <button type="button" onClick={addValue} style={gridAddBtnStyle}>
        + valor ao grid
      </button>
    );
  }

  return (
    <div
      style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 6 }}
    >
      <div style={gridAxisTagStyle}>grade · {values.length} valores</div>
      {extras.map((v, i) => {
        const err = validateGridValue(name, schema, v);
        return (
          <div key={i}>
            <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
              {enumOpts ? (
                <select
                  value={String(v)}
                  onChange={(e) => setExtra(i, e.target.value)}
                  style={gridRowControlStyle}
                >
                  {enumOpts.map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  inputMode="decimal"
                  value={String(v ?? "")}
                  onChange={(e) => setExtra(i, e.target.value)}
                  style={gridRowControlStyle}
                />
              )}
              <button
                type="button"
                onClick={() => removeExtra(i)}
                title="Remover valor"
                style={gridRemoveBtnStyle}
              >
                ×
              </button>
            </div>
            {err && <div style={gridErrStyle}>{err}</div>}
          </div>
        );
      })}
      <button type="button" onClick={addValue} style={gridAddBtnStyle}>
        + valor
      </button>
    </div>
  );
}

/** Banner shown when the grid_search block is active: explains the inline "+"
 * workflow and previews the Cartesian trial count from the active axes. */
function GridSearchBanner({
  hyperparameters,
}: {
  hyperparameters: Record<string, unknown[]>;
}) {
  const axes = Object.entries(hyperparameters).filter(
    ([, v]) => Array.isArray(v) && v.length > 1,
  );
  const totalTrials = axes.reduce(
    (acc, [, v]) => acc * v.length,
    axes.length ? 1 : 0,
  );
  return (
    <div style={gridBannerStyle}>
      <div
        style={{ fontFamily: "var(--font-mono)", fontSize: 11, lineHeight: 1.6 }}
      >
        <strong>Grid search ativo.</strong> Clique em{" "}
        <code>+ valor ao grid</code> nos hiperparâmetros de <em>Modelo</em> e{" "}
        <em>Treinamento</em> para varrer múltiplos valores.
      </div>
      {axes.length > 0 && (
        <div
          style={{
            marginTop: 8,
            display: "flex",
            flexDirection: "column",
            gap: 4,
          }}
        >
          <div style={gridAxisTagStyle}>
            // {totalTrials} trial{totalTrials === 1 ? "" : "s"}
            {totalTrials > 12 ? " ⚠️ alto" : ""}
          </div>
          {axes.map(([k, v]) => (
            <div
              key={k}
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--vf-text-dim)",
              }}
            >
              <code>{k}</code>: {v.map(String).join(", ")}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


/** Random search editor — dot-path keys mapped to (type, params) rows.
 *
 * Each row picks one of the three param types accepted by the backend:
 * - uniform(low, high)
 * - log_uniform(low, high)   ← natural for LR and weight_decay
 * - choice(options)          ← CSV-typed list, smart-coerced to int/float/bool
 *
 * The output shape matches what ``visionforge.blocks.random_search`` expects:
 * a dict from dot-path → {type, ...}. ``n_trials`` controls how many
 * configurations are sampled before the search ends.
 */
type RandomParamType = "uniform" | "log_uniform" | "choice";

interface RandomParamDef {
  type: RandomParamType;
  low?: number;
  high?: number;
  options?: Array<number | boolean | string>;
}

function defaultRandomParam(type: RandomParamType): RandomParamDef {
  if (type === "choice") return { type, options: [] };
  if (type === "log_uniform") return { type, low: 1e-5, high: 1e-1 };
  return { type, low: 0, high: 1 };
}

function RandomSearchFields({
  value,
  onChange,
}: {
  value: Record<string, unknown>;
  onChange: (v: Record<string, unknown>) => void;
}) {
  const n_trials = typeof value["n_trials"] === "number"
    ? (value["n_trials"] as number)
    : 10;
  const seed = typeof value["seed"] === "number" ? (value["seed"] as number) : 42;
  const space = (value["search_space"] ?? {}) as Record<string, RandomParamDef>;
  const entries = Object.entries(space);

  const setSpace = (next: Record<string, RandomParamDef>) =>
    onChange({ ...value, search_space: next });

  const renameKey = (oldKey: string, newKey: string) => {
    if (oldKey === newKey) return;
    if (newKey in space && newKey !== oldKey) return;
    const next: Record<string, RandomParamDef> = {};
    for (const [k, v] of entries) {
      next[k === oldKey ? newKey : k] = v;
    }
    setSpace(next);
  };

  const setParamType = (key: string, type: RandomParamType) => {
    setSpace({ ...space, [key]: defaultRandomParam(type) });
  };

  const setParamField = (
    key: string,
    field: "low" | "high",
    raw: string,
  ) => {
    const n = parseFloat(raw);
    if (Number.isNaN(n)) return;
    setSpace({ ...space, [key]: { ...space[key], [field]: n } });
  };

  const setChoiceOptions = (key: string, csv: string) => {
    setSpace({ ...space, [key]: { ...space[key], options: parseGridValuesCsv(csv) } });
  };

  const addRow = () => {
    let suffix = entries.length + 1;
    let key = `param_${suffix}`;
    while (key in space) {
      suffix += 1;
      key = `param_${suffix}`;
    }
    setSpace({ ...space, [key]: defaultRandomParam("uniform") });
  };

  const removeRow = (key: string) => {
    const next = { ...space };
    delete next[key];
    setSpace(next);
  };

  return (
    <div
      style={{
        marginBottom: 26,
        padding: 18,
        background: "rgba(255,255,255,0.02)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 12,
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 12,
        }}
      >
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            letterSpacing: "0.20em",
            textTransform: "uppercase",
            color: "var(--vf-text-muted)",
          }}
        >
          // random search · {n_trials} trial{n_trials === 1 ? "" : "s"}
        </div>
        <button
          type="button"
          onClick={addRow}
          style={{
            padding: "6px 12px",
            background: "var(--accent-soft)",
            border: "1px solid var(--accent-vf)",
            borderRadius: 8,
            color: "var(--vf-text)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            letterSpacing: "0.10em",
            textTransform: "uppercase",
            cursor: "pointer",
          }}
        >
          + adicionar
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <NumberField
          label="n_trials"
          value={n_trials}
          onChange={(v) => onChange({ ...value, n_trials: Math.max(1, Math.round(v)) })}
          min={1}
          step={1}
        />
        <NumberField
          label="seed"
          value={seed}
          onChange={(v) => onChange({ ...value, seed: Math.max(0, Math.round(v)) })}
          min={0}
          step={1}
        />
      </div>

      {entries.length === 0 ? (
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
          Espaço de busca vazio — clique em "+ adicionar".
          <div style={{ fontSize: 10, marginTop: 4, opacity: 0.7 }}>
            Exemplo: <code>training.learning_rate</code> = log_uniform(1e-5, 1e-2).
          </div>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {entries.map(([key, def]) => (
            <RandomSearchRow
              key={key}
              paramKey={key}
              def={def}
              onRenameKey={(nk) => renameKey(key, nk)}
              onChangeType={(t) => setParamType(key, t)}
              onChangeField={(field, raw) => setParamField(key, field, raw)}
              onChangeChoices={(csv) => setChoiceOptions(key, csv)}
              onRemove={() => removeRow(key)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function RandomSearchRow({
  paramKey,
  def,
  onRenameKey,
  onChangeType,
  onChangeField,
  onChangeChoices,
  onRemove,
}: {
  paramKey: string;
  def: RandomParamDef;
  onRenameKey: (k: string) => void;
  onChangeType: (t: RandomParamType) => void;
  onChangeField: (field: "low" | "high", raw: string) => void;
  onChangeChoices: (csv: string) => void;
  onRemove: () => void;
}) {
  const isChoice = def.type === "choice";
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: isChoice ? "1fr 130px 2fr auto" : "1fr 130px 1fr 1fr auto",
        gap: 8,
        alignItems: "center",
      }}
    >
      <input
        type="text"
        value={paramKey}
        onChange={(e) => onRenameKey(e.target.value)}
        placeholder="dot-path (ex: training.learning_rate)"
        style={rsInputStyle}
      />
      <select
        value={def.type}
        onChange={(e) => onChangeType(e.target.value as RandomParamType)}
        style={rsInputStyle}
      >
        <option value="uniform">uniform</option>
        <option value="log_uniform">log_uniform</option>
        <option value="choice">choice</option>
      </select>
      {isChoice ? (
        <input
          type="text"
          value={
            Array.isArray(def.options) ? def.options.map(String).join(", ") : ""
          }
          onChange={(e) => onChangeChoices(e.target.value)}
          placeholder="csv: resnet18, resnet50"
          style={rsInputStyle}
        />
      ) : (
        <>
          <input
            type="number"
            value={def.low ?? 0}
            step="any"
            onChange={(e) => onChangeField("low", e.target.value)}
            placeholder="low"
            style={rsInputStyle}
          />
          <input
            type="number"
            value={def.high ?? 1}
            step="any"
            onChange={(e) => onChangeField("high", e.target.value)}
            placeholder="high"
            style={rsInputStyle}
          />
        </>
      )}
      <button
        type="button"
        onClick={onRemove}
        title="Remover linha"
        style={{
          width: 32,
          height: 32,
          padding: 0,
          background: "transparent",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 8,
          color: "var(--vf-text-dim)",
          fontFamily: "var(--font-mono)",
          fontSize: 14,
          cursor: "pointer",
        }}
      >
        ×
      </button>
    </div>
  );
}

const rsInputStyle: React.CSSProperties = {
  padding: "8px 12px",
  background: "rgba(0,0,0,0.30)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 8,
  fontFamily: "var(--font-mono)",
  fontSize: 12,
  color: "var(--vf-text)",
};


/** ModelComparison hyperparameters — model_names multi-select + ranking metric.
 *
 * The list of available architectures comes from the JSON Schema for
 * ``ModelComparisonConfig.model_names.items.enum`` so it stays in sync with
 * the backend's Literal type without duplication.
 */
function ModelComparisonFields({
  schema,
  defs,
  value,
  onChange,
  errors,
}: {
  schema: JsonSchema;
  defs: Record<string, JsonSchema>;
  value: Record<string, unknown>;
  onChange: (v: Record<string, unknown>) => void;
  errors: ValidationError[];
}) {
  const resolved = resolveSchema(schema, defs);
  if (!resolved.properties) return null;

  const namesSchema = resolved.properties["model_names"];
  const namesResolved = namesSchema ? resolveSchema(namesSchema, defs) : null;
  const itemEnum =
    namesResolved?.items && "enum" in namesResolved.items
      ? (namesResolved.items.enum as string[] | undefined)
      : undefined;
  const archOptions = itemEnum ?? [];

  const selectedNames = Array.isArray(value["model_names"])
    ? (value["model_names"] as string[])
    : [];
  const metric = String(value["metric"] ?? "f1");

  const toggleArch = (arch: string) => {
    const next = selectedNames.includes(arch)
      ? selectedNames.filter((a) => a !== arch)
      : [...selectedNames, arch];
    onChange({ ...value, model_names: next });
  };

  const setMetric = (v: string) => onChange({ ...value, metric: v });

  const namesErr = errors.find(
    (e) =>
      e.field.length === 2 &&
      e.field[0] === "model_comparison" &&
      e.field[1] === "model_names",
  );

  return (
    <div
      style={{
        marginBottom: 26,
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
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            letterSpacing: "0.20em",
            textTransform: "uppercase",
            color: "var(--vf-text-muted)",
          }}
        >
          // comparação de modelos · {selectedNames.length} selecionado
          {selectedNames.length === 1 ? "" : "s"}
        </div>
        <div style={{ minWidth: 200 }}>
          {resolved.properties["metric"] && (
            <SchemaFieldVF
              name="metric"
              schema={resolved.properties["metric"]}
              defs={defs}
              value={metric}
              onChange={(v) => setMetric(String(v))}
              errors={errors}
              path={["model_comparison", "metric"]}
            />
          )}
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
          gap: 8,
        }}
      >
        {archOptions.map((arch) => {
          const selected = selectedNames.includes(arch);
          return (
            <label
              key={arch}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "8px 12px",
                background: selected
                  ? "var(--accent-soft)"
                  : "rgba(0,0,0,0.30)",
                border: `1px solid ${selected ? "var(--accent-vf)" : "var(--vf-panel-stroke)"}`,
                borderRadius: 8,
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                color: selected ? "var(--vf-text)" : "var(--vf-text-dim)",
                cursor: "pointer",
                transition: "background 160ms ease, color 160ms ease",
              }}
            >
              <input
                type="checkbox"
                checked={selected}
                onChange={() => toggleArch(arch)}
                style={{ accentColor: "var(--accent-vf)" }}
              />
              {arch}
            </label>
          );
        })}
      </div>

      {namesErr && (
        <p
          style={{
            fontSize: 11,
            color: "oklch(0.704 0.191 22.216)",
            fontFamily: "var(--font-mono)",
            margin: 0,
          }}
        >
          {namesErr.message}
        </p>
      )}
      {selectedNames.length < 2 && (
        <p
          style={{
            fontSize: 10,
            color: "var(--vf-text-muted)",
            fontFamily: "var(--font-mono)",
            margin: 0,
            fontStyle: "italic",
          }}
        >
          Selecione pelo menos 2 arquiteturas para iniciar a comparação.
        </p>
      )}
    </div>
  );
}


/** Transfer-learning hyperparameters — mode, unfreeze_from_layer, backbone_lr_multiplier. */
function TransferLearningFields({
  schema,
  defs,
  value,
  onChange,
  errors,
}: {
  schema: JsonSchema;
  defs: Record<string, JsonSchema>;
  value: Record<string, unknown>;
  onChange: (v: Record<string, unknown>) => void;
  errors: ValidationError[];
}) {
  const resolved = resolveSchema(schema, defs);
  if (!resolved.properties) return null;

  const setParam = (k: string, v: unknown) => onChange({ ...value, [k]: v });
  const mode = String(value["mode"] ?? "feature_extraction");

  // unfreeze_from_layer only applies in fine_tuning mode; backbone_lr_multiplier
  // is most useful in fine_tuning but also valid in feature_extraction (no-op).
  const showLayer = mode === "fine_tuning";

  return (
    <div
      style={{
        marginBottom: 26,
        padding: 18,
        background: "rgba(255,255,255,0.02)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 12,
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.20em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
        }}
      >
        // transfer learning
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: showLayer
            ? "repeat(3, 1fr)"
            : "repeat(2, 1fr)",
          gap: 14,
        }}
      >
        {resolved.properties["mode"] && (
          <SchemaFieldVF
            name="mode"
            schema={resolved.properties["mode"]}
            defs={defs}
            value={mode}
            onChange={(v) => setParam("mode", v)}
            errors={errors}
            path={["transfer_learning", "mode"]}
          />
        )}
        {showLayer && resolved.properties["unfreeze_from_layer"] && (
          <SchemaFieldVF
            name="unfreeze_from_layer"
            schema={resolved.properties["unfreeze_from_layer"]}
            defs={defs}
            value={value["unfreeze_from_layer"]}
            onChange={(v) =>
              setParam("unfreeze_from_layer", v === "" ? null : v)
            }
            errors={errors}
            path={["transfer_learning", "unfreeze_from_layer"]}
          />
        )}
        {resolved.properties["backbone_lr_multiplier"] && (
          <SchemaFieldVF
            name="backbone_lr_multiplier"
            schema={resolved.properties["backbone_lr_multiplier"]}
            defs={defs}
            value={value["backbone_lr_multiplier"]}
            onChange={(v) => setParam("backbone_lr_multiplier", v)}
            errors={errors}
            path={["transfer_learning", "backbone_lr_multiplier"]}
          />
        )}
      </div>
    </div>
  );
}


/** Read-only display for num_classes when task=binary forces it to 1. */
function LockedNumClasses() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 9,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
        }}
      >
        Nº de classes
      </span>
      <div
        title="Tarefa binária — fixado em 1"
        style={{
          padding: "8px 12px",
          background: "rgba(0,0,0,0.35)",
          border: "1px dashed var(--vf-panel-stroke)",
          borderRadius: 8,
          fontFamily: "var(--font-mono)",
          fontSize: 13,
          color: "var(--vf-text-dim)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          opacity: 0.85,
        }}
      >
        <span>1</span>
        <span
          style={{
            fontSize: 9,
            letterSpacing: "0.14em",
            color: "var(--vf-text-muted)",
            textTransform: "uppercase",
          }}
        >
          🔒 binário
        </span>
      </div>
    </div>
  );
}


interface FieldProps {
  name: string;
  schema: JsonSchema;
  defs: Record<string, JsonSchema>;
  value: unknown;
  onChange: (v: unknown) => void;
  errors: ValidationError[];
  path: string[];
}

/** Render a single schema field using the VisionForge design-system controls. */
function SchemaFieldVF({
  name,
  schema,
  defs,
  value,
  onChange,
  errors,
  path,
}: FieldProps) {
  const resolved = resolveSchema(schema, defs);
  const kind = resolveKind(name, resolved);
  // Path-qualified labels win over leaf-name labels: `model.name` is the
  // architecture, not the experiment name.
  const label =
    FIELD_LABELS[path.join(".")] ?? FIELD_LABELS[name] ?? resolved.title ?? name;
  const errorMsg = errors.find(
    (e) =>
      e.field.length === path.length && e.field.every((f, i) => f === path[i]),
  )?.message;

  const grid = useContext(GridContext);
  const dotPath = path.join(".");
  const gridable = grid.enabled && isGridableField(name, kind);

  // When this field is an active grid axis, keep value #0 of the axis in sync
  // with the primary control so the scalar config stays valid for everything else.
  const handleChange = (v: unknown) => {
    onChange(v);
    if (gridable) {
      const axis = grid.hyperparameters[dotPath];
      if (Array.isArray(axis) && axis.length) {
        const next = [...axis];
        next[0] = v;
        grid.setAxis(dotPath, next);
      }
    }
  };

  const gridExt = gridable ? (
    <GridAxisExtension
      dotPath={dotPath}
      name={name}
      schema={resolved}
      primaryValue={value}
    />
  ) : null;

  if (kind === "skip") return null;

  if (kind === "object") {
    // Recurse into nested object
    const inner = resolveSchema(schema, defs);
    if (inner.type !== "object" || !inner.properties) return null;
    const objVal = (value ?? {}) as Record<string, unknown>;
    return (
      <div
        style={{
          padding: 18,
          background: "rgba(255,255,255,0.015)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 12,
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
          gap: 16,
        }}
      >
        <div
          style={{
            gridColumn: "1 / -1",
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            color: "var(--vf-text-muted)",
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            marginBottom: 4,
          }}
        >
          {SECTION_LABELS[name] ?? resolved.title ?? name}
        </div>
        {Object.entries(inner.properties).map(([key, propSchema]) => (
          <SchemaFieldVF
            key={key}
            name={key}
            schema={propSchema}
            defs={defs}
            value={objVal[key]}
            onChange={(v) => onChange({ ...objVal, [key]: v })}
            errors={errors}
            path={[...path, key]}
          />
        ))}
      </div>
    );
  }

  if (kind === "select") {
    const opts = (resolved.enum ?? []).map((e) => ({
      value: String(e),
      label: String(e),
    }));
    return (
      <div>
        <SelectField
          label={label}
          value={String(value ?? resolved.default ?? resolved.enum?.[0] ?? "")}
          onChange={(v) => handleChange(v)}
          options={opts}
        />
        {errorMsg && (
          <p
            style={{
              fontSize: 11,
              color: "oklch(0.704 0.191 22.216)",
              marginTop: 4,
              fontFamily: "var(--font-mono)",
            }}
          >
            {errorMsg}
          </p>
        )}
        {gridExt}
      </div>
    );
  }

  if (kind === "segmented") {
    const opts = (resolved.enum ?? []).map((e) => ({
      value: String(e),
      label: String(e),
    }));
    return (
      <div>
        <Segmented
          label={label}
          value={String(value ?? resolved.default ?? resolved.enum?.[0] ?? "")}
          onChange={(v) => handleChange(v)}
          options={opts}
        />
        {errorMsg && (
          <p
            style={{
              fontSize: 11,
              color: "oklch(0.704 0.191 22.216)",
              marginTop: 4,
              fontFamily: "var(--font-mono)",
            }}
          >
            {errorMsg}
          </p>
        )}
        {gridExt}
      </div>
    );
  }

  if (kind === "toggle") {
    return (
      <div>
        <Toggle
          label={label}
          value={Boolean(value)}
          onChange={(v) => onChange(v)}
        />
        {errorMsg && (
          <p
            style={{
              fontSize: 11,
              color: "oklch(0.704 0.191 22.216)",
              marginTop: 4,
              fontFamily: "var(--font-mono)",
            }}
          >
            {errorMsg}
          </p>
        )}
      </div>
    );
  }

  if (kind === "number") {
    return (
      <div>
        <NumberField
          label={label}
          value={(value as number) ?? 0}
          onChange={(v) => handleChange(v)}
          min={resolved.minimum ?? resolved.exclusiveMinimum}
          step={resolved.type === "integer" ? 1 : undefined}
        />
        {errorMsg && (
          <p
            style={{
              fontSize: 11,
              color: "oklch(0.704 0.191 22.216)",
              marginTop: 4,
              fontFamily: "var(--font-mono)",
            }}
          >
            {errorMsg}
          </p>
        )}
        {gridExt}
      </div>
    );
  }

  if (kind === "array-number") {
    const arrVal = Array.isArray(value) ? value : [];
    return (
      <div>
        <TextField
          label={label}
          value={arrVal.join(", ")}
          onChange={(v) => {
            const parts = v
              .split(",")
              .map((s) => s.trim())
              .filter((s) => s !== "")
              .map(Number);
            onChange(parts);
          }}
          placeholder="ex: 0.485, 0.456, 0.406"
          mono
        />
        {errorMsg && (
          <p
            style={{
              fontSize: 11,
              color: "oklch(0.704 0.191 22.216)",
              marginTop: 4,
              fontFamily: "var(--font-mono)",
            }}
          >
            {errorMsg}
          </p>
        )}
      </div>
    );
  }

  // text
  return (
    <div>
      <TextField
        label={label}
        value={String(value ?? "")}
        onChange={(v) => onChange(v)}
        mono={name === "base_dir"}
      />
      {errorMsg && (
        <p
          style={{
            fontSize: 11,
            color: "oklch(0.704 0.191 22.216)",
            marginTop: 4,
            fontFamily: "var(--font-mono)",
          }}
        >
          {errorMsg}
        </p>
      )}
    </div>
  );
}

const yamlBtnStyle: React.CSSProperties = {
  padding: "8px 14px",
  background: "var(--accent-soft)",
  border: "1px solid var(--accent-vf)",
  borderRadius: 10,
  color: "var(--vf-text)",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  letterSpacing: "0.10em",
  textTransform: "uppercase" as const,
  cursor: "pointer",
  whiteSpace: "nowrap" as const,
  lineHeight: 1,
};

const yamlBtnSecondaryStyle: React.CSSProperties = {
  ...yamlBtnStyle,
  background: "transparent",
  border: "1px solid var(--vf-panel-stroke)",
  color: "var(--vf-text-dim)",
};

/** Glass card container for classification parameters. */
export function ParamPanel({
  task,
  schema,
  formData,
  setFormData,
  validationErrors,
}: ParamPanelProps) {
  const [importError, setImportError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleExport = () => {
    const expName = (formData["name"] as string) || "config";
    exportConfigToYaml(formData, expName);
  };

  const handleImportClick = () => {
    setImportError(null);
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    // Reset input so the same file can be re-imported if needed
    e.target.value = "";
    void importConfigFromYaml(file).then((result) => {
      if ("error" in result) {
        setImportError(result.error);
        return;
      }
      // Always load the parsed data so the user sees it in the form; if any
      // structural issues exist, warn them ahead of the submit round-trip.
      setFormData(result.data);
      if (!schema) {
        setImportError(null);
        return;
      }
      const issues = validateParsedConfig(
        result.data,
        schema,
        schema.$defs ?? {},
      );
      if (issues.length === 0) {
        setImportError(null);
      } else {
        const summary = issues
          .slice(0, 5)
          .map(
            (iss) => `· ${humanizeFieldPath(iss.field)} — ${iss.message}`,
          )
          .join("\n");
        const more =
          issues.length > 5 ? `\n…(+${issues.length - 5} mais)` : "";
        setImportError(
          `YAML importado com ${issues.length} aviso(s) estrutural(is):\n${summary}${more}\n\nCorrija antes de treinar — o backend rejeita por validação Pydantic.`,
        );
      }
    });
  };

  if (task.key !== "classification") {
    return (
      <section
        style={{
          position: "relative",
          padding: 48,
          background: "rgba(10,12,16,0.55)",
          backdropFilter: "blur(20px) saturate(140%)",
          WebkitBackdropFilter: "blur(20px) saturate(140%)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 20,
          textAlign: "center",
        }}
      >
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            color: "var(--vf-text-muted)",
            letterSpacing: "0.22em",
            textTransform: "uppercase",
            marginBottom: 12,
          }}
        >
          // em breve
        </div>
        <div
          style={{
            fontSize: 22,
            fontWeight: 600,
            color: "var(--vf-text-dim)",
          }}
        >
          {task.label} ainda não está disponível
        </div>
        <p
          style={{
            color: "var(--vf-text-muted)",
            fontSize: 14,
            marginTop: 8,
          }}
        >
          Esta tarefa será implementada em uma próxima fase do VisionForge.
        </p>
      </section>
    );
  }

  if (!schema) {
    return (
      <section
        style={{
          padding: 28,
          background: "rgba(10,12,16,0.55)",
          backdropFilter: "blur(20px) saturate(140%)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 20,
          textAlign: "center",
          color: "var(--vf-text-muted)",
          fontFamily: "var(--font-mono)",
          fontSize: 13,
          letterSpacing: "0.08em",
        }}
      >
        carregando schema…
      </section>
    );
  }

  const defs = schema.$defs ?? {};

  // Helper for nested field update
  const setField = (section: string, field: string, val: unknown) => {
    setFormData((prev) => {
      const sec = (prev[section] ?? {}) as Record<string, unknown>;
      return { ...prev, [section]: { ...sec, [field]: val } };
    });
  };

  // Model section from schema
  const modelSchema =
    schema.properties?.["model"] &&
    resolveSchema(schema.properties["model"], defs);
  const modelProps = modelSchema?.properties ?? {};

  // Training section from schema
  const trainingSchema =
    schema.properties?.["training"] &&
    resolveSchema(schema.properties["training"], defs);
  const trainingProps = trainingSchema?.properties ?? {};

  // Data section from schema
  const dataSchema =
    schema.properties?.["data"] &&
    resolveSchema(schema.properties["data"], defs);
  const dataProps = dataSchema?.properties ?? {};

  const modelData = (formData["model"] ?? {}) as Record<string, unknown>;
  const trainingData = (formData["training"] ?? {}) as Record<string, unknown>;
  const dataData = (formData["data"] ?? {}) as Record<string, unknown>;

  // Grid search axis state, shared with every SchemaFieldVF via context so each
  // gridable field can render its "+ valor ao grid" affordance.
  const gridEnabled = formData["block"] === "grid_search";
  const gridHyperparameters = (((formData["grid_search"] ??
    {}) as Record<string, unknown>)["hyperparameters"] ?? {}) as Record<
    string,
    unknown[]
  >;
  const gridCtx: GridContextValue = {
    enabled: gridEnabled,
    hyperparameters: gridHyperparameters,
    setAxis: (dotPath, values) =>
      setFormData((prev) => {
        const gs = (prev["grid_search"] ?? {}) as Record<string, unknown>;
        const hp = {
          ...((gs["hyperparameters"] ?? {}) as Record<string, unknown>),
        };
        if (values === null) delete hp[dotPath];
        else hp[dotPath] = values;
        return { ...prev, grid_search: { ...gs, hyperparameters: hp } };
      }),
  };

  return (
    <GridContext.Provider value={gridCtx}>
    <section
      key={task.key}
      style={{
        position: "relative",
        padding: 28,
        background: "rgba(10,12,16,0.55)",
        backdropFilter: "blur(20px) saturate(140%)",
        WebkitBackdropFilter: "blur(20px) saturate(140%)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 20,
        boxShadow:
          "0 30px 80px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04)",
        overflow: "hidden",
        animation: "fadeUp 360ms ease forwards",
      }}
    >
      {/* Accent glow corner */}
      <span
        style={{
          position: "absolute",
          top: -80,
          right: -80,
          width: 280,
          height: 280,
          borderRadius: "50%",
          background: "var(--accent-soft)",
          filter: "blur(40px)",
          pointerEvents: "none",
        }}
      />

      {/* Hidden file input for YAML import */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".yaml,.yml"
        style={{ display: "none" }}
        onChange={handleFileChange}
      />

      {/* Experiment name + task top row + YAML action buttons */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr auto",
          gap: 18,
          marginBottom: importError ? 10 : 26,
          position: "relative",
          alignItems: "end",
        }}
      >
        {schema.properties?.["name"] && (
          <SchemaFieldVF
            name="name"
            schema={schema.properties["name"]}
            defs={defs}
            value={formData["name"]}
            onChange={(v) =>
              setFormData((prev) => ({ ...prev, name: v }))
            }
            errors={validationErrors}
            path={["name"]}
          />
        )}
        {/* task (binário/multiclass) e num_classes vivem agora na seção
            // dataset → // classes, perto da detecção de classes. */}
        {/* YAML export / import buttons */}
        <div style={{ display: "flex", gap: 8, paddingBottom: 2 }}>
          <button
            type="button"
            onClick={handleExport}
            style={yamlBtnStyle}
            title="Exportar configuração atual como arquivo .yaml"
          >
            ↓ Exportar YAML
          </button>
          <button
            type="button"
            onClick={handleImportClick}
            style={yamlBtnSecondaryStyle}
            title="Importar configuração a partir de um arquivo .yaml"
          >
            ↑ Importar YAML
          </button>
        </div>
      </div>

      {/* Import error banner */}
      {importError && (
        <div
          style={{
            marginBottom: 18,
            padding: "10px 14px",
            background: "oklch(0.704 0.191 22.216 / 0.10)",
            border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
            borderRadius: 10,
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            color: "oklch(0.85 0.14 22)",
            whiteSpace: "pre-line",
            lineHeight: 1.6,
          }}
        >
          {importError}
        </div>
      )}

      {/* Divider */}
      <div
        style={{
          height: 1,
          width: "100%",
          background:
            "linear-gradient(90deg, transparent, var(--vf-panel-stroke), transparent)",
          marginBottom: 26,
        }}
      />

      {/* Block selector — toggles classification / CV / transfer learning */}
      <BlockSelector
        value={String(formData["block"] ?? "classification")}
        onChange={(v) =>
          setFormData((prev) => {
            // Auto-populate sensible defaults so the field block below has
            // something to render. Backend Pydantic does the rest of the
            // validation at submit time.
            const next: Record<string, unknown> = { ...prev, block: v };
            if (v === "cross_validation" && !prev["cross_validation"]) {
              next["cross_validation"] = {
                n_folds: 5,
                stratified: true,
                shuffle: true,
                fold_seed: 42,
              };
            }
            if (v === "transfer_learning" && !prev["transfer_learning"]) {
              next["transfer_learning"] = {
                mode: "feature_extraction",
                unfreeze_from_layer: null,
                backbone_lr_multiplier: 0.1,
              };
            }
            if (v === "model_comparison" && !prev["model_comparison"]) {
              // Default to two light architectures so the user only has to
              // confirm or extend the list — never an empty selection that
              // would block submit.
              next["model_comparison"] = {
                model_names: ["resnet18", "resnet50"],
                metric: "f1",
              };
            }
            if (v === "grid_search" && !prev["grid_search"]) {
              // Axes are now built inline: the user clicks "+ valor ao grid" on
              // the model/training fields. Start with an empty search space.
              next["grid_search"] = { hyperparameters: {} };
            }
            if (v === "random_search" && !prev["random_search"]) {
              // Default search_space exercises the two most common axes —
              // LR (log_uniform) and weight_decay (log_uniform). Cheap to
              // edit, immediately runnable.
              next["random_search"] = {
                n_trials: 10,
                seed: 42,
                search_space: {
                  "training.learning_rate": {
                    type: "log_uniform",
                    low: 1e-5,
                    high: 1e-2,
                  },
                  "training.weight_decay": {
                    type: "log_uniform",
                    low: 1e-6,
                    high: 1e-3,
                  },
                },
              };
            }
            return next;
          })
        }
      />

      {/* Cross-validation fields — only when the block is K-Fold */}
      {formData["block"] === "cross_validation" &&
        schema.properties?.["cross_validation"] && (
          <CrossValidationFields
            schema={schema.properties["cross_validation"]}
            defs={defs}
            value={
              (formData["cross_validation"] ?? {}) as Record<string, unknown>
            }
            onChange={(v) =>
              setFormData((prev) => ({ ...prev, cross_validation: v }))
            }
            errors={validationErrors}
          />
        )}

      {/* Transfer learning fields — only when the block is TL */}
      {formData["block"] === "transfer_learning" &&
        schema.properties?.["transfer_learning"] && (
          <TransferLearningFields
            schema={schema.properties["transfer_learning"]}
            defs={defs}
            value={
              (formData["transfer_learning"] ?? {}) as Record<string, unknown>
            }
            onChange={(v) =>
              setFormData((prev) => ({ ...prev, transfer_learning: v }))
            }
            errors={validationErrors}
          />
        )}

      {/* Model comparison fields — only when comparing architectures */}
      {formData["block"] === "model_comparison" &&
        schema.properties?.["model_comparison"] && (
          <ModelComparisonFields
            schema={schema.properties["model_comparison"]}
            defs={defs}
            value={
              (formData["model_comparison"] ?? {}) as Record<string, unknown>
            }
            onChange={(v) =>
              setFormData((prev) => ({ ...prev, model_comparison: v }))
            }
            errors={validationErrors}
          />
        )}

      {/* Grid search — inline "+" axis editing happens on the model/training
          fields below; this banner explains the workflow + previews trials. */}
      {formData["block"] === "grid_search" && (
        <GridSearchBanner hyperparameters={gridHyperparameters} />
      )}

      {/* Random search fields — only when block is random_search */}
      {formData["block"] === "random_search" && (
        <RandomSearchFields
          value={(formData["random_search"] ?? {}) as Record<string, unknown>}
          onChange={(v) =>
            setFormData((prev) => ({ ...prev, random_search: v }))
          }
        />
      )}

      {/* Model section */}
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--vf-text-muted)",
          letterSpacing: "0.20em",
          textTransform: "uppercase",
          marginBottom: 14,
        }}
      >
        // modelo
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1.5fr 1fr 1fr",
          gap: 18,
          marginBottom: 26,
          position: "relative",
        }}
      >
        {modelProps["name"] && (
          <SchemaFieldVF
            name="name"
            schema={modelProps["name"]}
            defs={defs}
            value={modelData["name"]}
            onChange={(v) => setField("model", "name", v)}
            errors={validationErrors}
            path={["model", "name"]}
          />
        )}
        {/* num_classes movido para // dataset → // classes */}
        {modelProps["pretrained"] && (
          <SchemaFieldVF
            name="pretrained"
            schema={modelProps["pretrained"]}
            defs={defs}
            value={modelData["pretrained"]}
            onChange={(v) => setField("model", "pretrained", v)}
            errors={validationErrors}
            path={["model", "pretrained"]}
          />
        )}
        <WeightsPathField
          value={String(modelData["weights_path"] ?? "")}
          onChange={(v) => setField("model", "weights_path", v === "" ? null : v)}
        />
      </div>

      {/* Divider */}
      <div
        style={{
          height: 1,
          width: "100%",
          background:
            "linear-gradient(90deg, transparent, var(--vf-panel-stroke), transparent)",
          marginBottom: 26,
        }}
      />

      {/* Training section */}
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--vf-text-muted)",
          letterSpacing: "0.20em",
          textTransform: "uppercase",
          marginBottom: 14,
        }}
      >
        // treinamento
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 18,
          marginBottom: 26,
          position: "relative",
        }}
      >
        {["epochs", "learning_rate", "batch_size", "optimizer", "early_stopping_patience", "weight_decay", "seed", "deterministic", "mixed_precision"].map(
          (key) =>
            trainingProps[key] ? (
              <SchemaFieldVF
                key={key}
                name={key}
                schema={trainingProps[key]}
                defs={defs}
                value={trainingData[key]}
                onChange={(v) => setField("training", key, v)}
                errors={validationErrors}
                path={["training", key]}
              />
            ) : null,
        )}
      </div>

      {/* Scheduler section — nested under training */}
      {trainingProps["scheduler"] && (
        <SchedulerFields
          schema={trainingProps["scheduler"]}
          defs={defs}
          value={(trainingData["scheduler"] ?? {}) as Record<string, unknown>}
          onChange={(v) => setField("training", "scheduler", v)}
          errors={validationErrors}
        />
      )}

      {/* Divider */}
      <div
        style={{
          height: 1,
          width: "100%",
          background:
            "linear-gradient(90deg, transparent, var(--vf-panel-stroke), transparent)",
          marginBottom: 26,
        }}
      />

      {/* Data section */}
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--vf-text-muted)",
          letterSpacing: "0.20em",
          textTransform: "uppercase",
          marginBottom: 14,
        }}
      >
        // dataset
      </div>

      <DatasetPicker
        baseDir={(dataData["base_dir"] as string) ?? ""}
        trainDir={(dataData["train_dir"] as string) ?? ""}
        valDir={(dataData["val_dir"] as string) ?? ""}
        testDir={(dataData["test_dir"] as string) ?? ""}
        onChange={(next) =>
          setFormData((prev) => {
            const sec = (prev["data"] ?? {}) as Record<string, unknown>;
            return { ...prev, data: { ...sec, ...next } };
          })
        }
      />

      <DatasetStats
        baseDir={(dataData["base_dir"] as string) ?? ""}
        trainDir={(dataData["train_dir"] as string) ?? ""}
        valDir={(dataData["val_dir"] as string) ?? ""}
        testDir={(dataData["test_dir"] as string) ?? ""}
        onApplyClasses={(n) =>
          setFormData((prev) => {
            // 2-class datasets → BCE head (num_classes=1, task=binary)
            // Anything else → multiclass with the detected count
            const model = (prev["model"] ?? {}) as Record<string, unknown>;
            const isBinary = n === 2;
            return {
              ...prev,
              task: isBinary ? "binary" : "multiclass",
              model: { ...model, num_classes: isBinary ? 1 : n },
            };
          })
        }
      />

      {/* Classes — task + nº de classes, alimentados pela detecção do dataset.
          Posicionados aqui (não no topo / em // modelo) porque o número de
          classes é uma propriedade do dataset selecionado acima. */}
      <div
        style={{
          marginTop: 16,
          marginBottom: 12,
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--vf-text-muted)",
          letterSpacing: "0.20em",
          textTransform: "uppercase",
        }}
      >
        // classes
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 18,
          marginBottom: 26,
        }}
      >
        {schema.properties?.["task"] && (
          <SchemaFieldVF
            name="task"
            schema={schema.properties["task"]}
            defs={defs}
            value={formData["task"]}
            onChange={(v) =>
              setFormData((prev) => {
                // Binary task is incompatible with num_classes >= 2; auto-lock
                // num_classes = 1 to mirror the Pydantic model_validator.
                const model = (prev["model"] ?? {}) as Record<string, unknown>;
                const nextNumClasses =
                  v === "binary"
                    ? 1
                    : (model["num_classes"] === 1 ? 2 : model["num_classes"]);
                return {
                  ...prev,
                  task: v,
                  model: { ...model, num_classes: nextNumClasses },
                };
              })
            }
            errors={validationErrors}
            path={["task"]}
          />
        )}
        {modelProps["num_classes"] &&
          (formData["task"] === "binary" ? (
            <LockedNumClasses />
          ) : (
            <SchemaFieldVF
              name="num_classes"
              schema={modelProps["num_classes"]}
              defs={defs}
              value={modelData["num_classes"]}
              onChange={(v) => setField("model", "num_classes", v)}
              errors={validationErrors}
              path={["model", "num_classes"]}
            />
          ))}
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 18,
          marginBottom: 16,
        }}
      >
        {dataProps["num_workers"] && (
          <NumWorkersField
            value={(dataData["num_workers"] as number) ?? 0}
            onChange={(v) => setField("data", "num_workers", v)}
          />
        )}
        {dataProps["pin_memory"] && (
          <SchemaFieldVF
            name="pin_memory"
            schema={dataProps["pin_memory"]}
            defs={defs}
            value={dataData["pin_memory"]}
            onChange={(v) => setField("data", "pin_memory", v)}
            errors={validationErrors}
            path={["data", "pin_memory"]}
          />
        )}
      </div>

      {/* Divider — separa os ajustes de DataLoader das transformações de imagem */}
      <div
        style={{
          height: 1,
          width: "100%",
          background:
            "linear-gradient(90deg, transparent, var(--vf-panel-stroke), transparent)",
          marginBottom: 26,
        }}
      />

      {/* Pré-processamento (filtros) — roda antes de augmentation/normalização */}
      <PreprocessingPanel
        baseDir={(dataData["base_dir"] as string) ?? ""}
        steps={toUIPreprocessingSteps(
          ((dataData["preprocessing"] ?? {}) as Record<string, unknown>)["steps"],
        )}
        onChange={(uiSteps) =>
          setFormData((prev) => {
            const sec = (prev["data"] ?? {}) as Record<string, unknown>;
            const pp = (sec["preprocessing"] ?? {}) as Record<string, unknown>;
            return {
              ...prev,
              data: {
                ...sec,
                preprocessing: {
                  ...pp,
                  steps: fromUIPreprocessingSteps(uiSteps),
                },
              },
            };
          })
        }
      />

      {/* Augmentation / Transforms sub-section */}
      {dataProps["transforms"] && (
        <>
          <div
            style={{
              height: 1,
              width: "100%",
              background:
                "linear-gradient(90deg, transparent, var(--vf-panel-stroke), transparent)",
              margin: "20px 0 14px",
            }}
          />
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "var(--vf-text-muted)",
              letterSpacing: "0.20em",
              textTransform: "uppercase",
              marginBottom: 14,
            }}
          >
            // aumentos &amp; normalização
          </div>
          <SchemaFieldVF
            name="transforms"
            schema={dataProps["transforms"]}
            defs={defs}
            value={dataData["transforms"]}
            onChange={(v) => setField("data", "transforms", v)}
            errors={validationErrors}
            path={["data", "transforms"]}
          />
          <AugmentPreview
            baseDir={(dataData["base_dir"] as string) ?? ""}
            transforms={(dataData["transforms"] ?? {}) as Record<string, unknown>}
          />
        </>
      )}

      {/* Validation errors summary */}
      {validationErrors.length > 0 && (
        <div
          style={{
            marginTop: 18,
            padding: "12px 16px",
            background: "oklch(0.704 0.191 22.216 / 0.10)",
            border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
            borderRadius: 10,
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            color: "oklch(0.85 0.14 22)",
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 8 }}>
            {validationErrors.length} campo(s) com erro:
          </div>
          <ul style={{ margin: 0, paddingLeft: 18, lineHeight: 1.65 }}>
            {validationErrors.map((err, i) => (
              <li key={i}>
                <span style={{ color: "var(--vf-text)" }}>
                  {humanizeFieldPath(err.field)}
                </span>
                <span style={{ color: "oklch(0.85 0.14 22)" }}>
                  {" — "}
                  {err.message}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
    </GridContext.Provider>
  );
}
