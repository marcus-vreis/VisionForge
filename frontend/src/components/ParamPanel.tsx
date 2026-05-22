import { useRef, useState } from "react";
import { humanizeFieldPath, type ValidationError } from "../hooks/useExperiment";
import type { JsonSchema } from "../types/schema";
import type { TaskDefinition } from "../types/tasks";
import { exportConfigToYaml, importConfigFromYaml } from "../lib/yaml-config";
import { DatasetPicker } from "./DatasetPicker";
import { DatasetStats } from "./DatasetStats";
import { resolveKind } from "./field-renderer";
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

/** Human-readable field labels. */
const FIELD_LABELS: Record<string, string> = {
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
  const label = FIELD_LABELS[name] ?? resolved.title ?? name;
  const errorMsg = errors.find(
    (e) =>
      e.field.length === path.length && e.field.every((f, i) => f === path[i]),
  )?.message;

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
          onChange={(v) => onChange(v)}
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
          onChange={(v) => onChange(v)}
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
          onChange={(v) => onChange(v)}
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
      } else {
        setImportError(null);
        setFormData(result.data);
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

  return (
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
          gridTemplateColumns: "1fr 1fr auto",
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
        {schema.properties?.["task"] && (
          <SchemaFieldVF
            name="task"
            schema={schema.properties["task"]}
            defs={defs}
            value={formData["task"]}
            onChange={(v) =>
              setFormData((prev) => ({ ...prev, task: v }))
            }
            errors={validationErrors}
            path={["task"]}
          />
        )}
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
        {modelProps["num_classes"] && (
          <SchemaFieldVF
            name="num_classes"
            schema={modelProps["num_classes"]}
            defs={defs}
            value={modelData["num_classes"]}
            onChange={(v) => setField("model", "num_classes", v)}
            errors={validationErrors}
            path={["model", "num_classes"]}
          />
        )}
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
        {["epochs", "learning_rate", "batch_size", "optimizer", "early_stopping_patience", "weight_decay", "seed"].map(
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
      />

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 18,
          marginBottom: 16,
        }}
      >
        {dataProps["num_workers"] && (
          <SchemaFieldVF
            name="num_workers"
            schema={dataProps["num_workers"]}
            defs={defs}
            value={dataData["num_workers"]}
            onChange={(v) => setField("data", "num_workers", v)}
            errors={validationErrors}
            path={["data", "num_workers"]}
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
  );
}
