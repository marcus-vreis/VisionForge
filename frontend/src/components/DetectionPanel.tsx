import { useState } from "react";
import { pickDatasetFolder } from "../api/client";
import {
  DETECTION_BACKENDS,
  DETECTION_MODELS,
  defaultModelForBackend,
  isValidModelForBackend,
  type DetectionBackend,
  type DetectionForm,
} from "../lib/detection-models";
import type { ValidationError } from "../hooks/useExperiment";
import { NumberField, SelectField, Segmented, TextField, Toggle } from "./controls";
import { DetectionDatasetStats } from "./DetectionDatasetStats";

interface DetectionPanelProps {
  formData: DetectionForm;
  setFormData: (updater: (prev: DetectionForm) => DetectionForm) => void;
  accent: string;
  validationErrors: ValidationError[];
}

const sectionLabel: React.CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  letterSpacing: "0.22em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
  marginBottom: 12,
};

const grid: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
  gap: 14,
};

/** Schema-aligned form for an Ultralytics/torchvision detection run. */
export function DetectionPanel({
  formData,
  setFormData,
  accent,
  validationErrors,
}: DetectionPanelProps) {
  const [picking, setPicking] = useState(false);

  const setModel = (patch: Partial<DetectionForm["model"]>) =>
    setFormData((p) => ({ ...p, model: { ...p.model, ...patch } }));
  const setData = (patch: Partial<DetectionForm["data"]>) =>
    setFormData((p) => ({ ...p, data: { ...p.data, ...patch } }));
  const setTraining = (patch: Partial<DetectionForm["training"]>) =>
    setFormData((p) => ({ ...p, training: { ...p.training, ...patch } }));

  const onBackendChange = (raw: string) => {
    const backend = raw as DetectionBackend;
    setFormData((p) => {
      const name = isValidModelForBackend(backend, p.model.name)
        ? p.model.name
        : defaultModelForBackend(backend);
      return { ...p, model: { ...p.model, backend, name } };
    });
  };

  const onPickFolder = async () => {
    setPicking(true);
    try {
      const res = await pickDatasetFolder();
      if (!res.cancelled && res.path) setData({ base_dir: res.path });
    } finally {
      setPicking(false);
    }
  };

  const card: React.CSSProperties = {
    background: "var(--vf-panel)",
    border: "1px solid var(--vf-panel-stroke)",
    borderRadius: 18,
    padding: 26,
    backdropFilter: "blur(14px)",
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      {validationErrors.length > 0 && (
        <div
          style={{
            padding: "12px 16px",
            background: "oklch(0.704 0.191 22.216 / 0.10)",
            border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
            borderRadius: 12,
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            color: "oklch(0.85 0.14 22)",
          }}
        >
          {validationErrors.slice(0, 5).map((e, i) => (
            <div key={i}>
              {e.field.join(" › ")}: {e.message}
            </div>
          ))}
        </div>
      )}

      {/* Modelo */}
      <div style={card}>
        <div style={sectionLabel}>Modelo · detecção</div>
        <div style={grid}>
          <Segmented
            label="Backend"
            value={formData.model.backend}
            onChange={onBackendChange}
            options={DETECTION_BACKENDS.map((b) => ({
              value: b,
              label: b === "ultralytics" ? "Ultralytics" : "Torchvision",
            }))}
            hint="origem do modelo"
          />
          <SelectField
            label="Arquitetura"
            value={formData.model.name}
            onChange={(v) => setModel({ name: v })}
            options={DETECTION_MODELS[formData.model.backend]}
            hint="detector"
          />
          <NumberField
            label="Nº de classes"
            value={formData.model.num_classes}
            onChange={(v) => setModel({ num_classes: Math.round(v) })}
            min={1}
            step={1}
            hint="classes do dataset"
          />
          <Toggle
            label="Pesos pré-treinados"
            value={formData.model.pretrained}
            onChange={(v) => setModel({ pretrained: v })}
            hint="COCO"
          />
        </div>
      </div>

      {/* Dataset */}
      <div style={card}>
        <div style={sectionLabel}>Dataset (layout YOLO)</div>
        <div style={grid}>
          <div style={{ gridColumn: "1 / -1", display: "flex", gap: 10, alignItems: "flex-end" }}>
            <div style={{ flex: 1 }}>
              <TextField
                label="Pasta base"
                value={formData.data.base_dir}
                onChange={(v) => setData({ base_dir: v })}
                placeholder="…/dataset (images/train, images/val)"
                hint="raiz YOLO"
                mono
              />
            </div>
            <button
              type="button"
              onClick={() => void onPickFolder()}
              disabled={picking}
              style={{
                padding: "12px 16px",
                background: "var(--accent-soft)",
                border: `1px solid ${accent}`,
                borderRadius: 10,
                color: "var(--vf-text)",
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                cursor: picking ? "default" : "pointer",
                whiteSpace: "nowrap",
              }}
            >
              {picking ? "…" : "📁 Escolher"}
            </button>
          </div>
          <NumberField
            label="Image size"
            value={formData.data.image_size}
            onChange={(v) => setData({ image_size: Math.round(v) })}
            min={32}
            step={32}
            suffix="px"
            hint="imgsz"
          />
        </div>
        <DetectionDatasetStats
          baseDir={formData.data.base_dir}
          onApplyClasses={(n) => setModel({ num_classes: n })}
        />
      </div>

      {/* Treinamento */}
      <div style={card}>
        <div style={sectionLabel}>Treinamento</div>
        <div style={grid}>
          <NumberField
            label="Épocas"
            value={formData.training.epochs}
            onChange={(v) => setTraining({ epochs: Math.round(v) })}
            min={1}
            step={1}
          />
          <NumberField
            label="Batch size"
            value={formData.training.batch_size}
            onChange={(v) => setTraining({ batch_size: Math.round(v) })}
            min={1}
            step={1}
            hint="qualquer inteiro"
          />
          <NumberField
            label="Learning rate"
            value={formData.training.learning_rate}
            onChange={(v) => setTraining({ learning_rate: v })}
            min={0.000001}
            step={0.001}
            hint="lr0"
          />
          <NumberField
            label="Patience"
            value={formData.training.patience}
            onChange={(v) => setTraining({ patience: Math.round(v) })}
            min={0}
            step={1}
            hint="early stop"
          />
          <NumberField
            label="Seed"
            value={formData.training.seed}
            onChange={(v) => setTraining({ seed: Math.round(v) })}
            min={0}
            step={1}
          />
          <NumberField
            label="Workers"
            value={formData.training.workers}
            onChange={(v) => setTraining({ workers: Math.round(v) })}
            min={0}
            step={1}
          />
        </div>
      </div>
    </div>
  );
}
