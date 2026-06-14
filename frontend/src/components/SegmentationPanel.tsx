import { useState } from "react";
import { pickDatasetFolder } from "../api/client";
import {
  SEGMENTATION_LOSSES,
  SEGMENTATION_MODELS,
  ignoreIndexCollides,
  type SegmentationForm,
} from "../lib/segmentation-models";
import type { ValidationError } from "../hooks/useExperiment";
import { NumberField, SelectField, Segmented, TextField, Toggle } from "./controls";
import { ComparisonCard } from "./ComparisonCard";
import { SweepCard, type SweepPayload } from "./SweepCard";

const COMPARE_METRICS = [
  { value: "miou", label: "mIoU" },
  { value: "dice", label: "Dice" },
  { value: "pixel_acc", label: "Pixel acc." },
];

const SWEEP_PATH_HINTS = [
  "training.learning_rate",
  "training.batch_size",
  "model.name",
];

interface SegmentationPanelProps {
  formData: SegmentationForm;
  setFormData: (updater: (prev: SegmentationForm) => SegmentationForm) => void;
  accent: string;
  validationErrors: ValidationError[];
  busy?: boolean;
  onCompare?: (modelNames: string[], metric: string) => void;
  onSweep?: (payload: SweepPayload) => void;
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

/** Schema-aligned form for a paired image/mask semantic-segmentation run. */
export function SegmentationPanel({
  formData,
  setFormData,
  accent,
  validationErrors,
  busy,
  onCompare,
  onSweep,
}: SegmentationPanelProps) {
  const [picking, setPicking] = useState(false);

  const setModel = (patch: Partial<SegmentationForm["model"]>) =>
    setFormData((p) => ({ ...p, model: { ...p.model, ...patch } }));
  const setData = (patch: Partial<SegmentationForm["data"]>) =>
    setFormData((p) => ({ ...p, data: { ...p.data, ...patch } }));
  const setTraining = (patch: Partial<SegmentationForm["training"]>) =>
    setFormData((p) => ({ ...p, training: { ...p.training, ...patch } }));

  const onPickFolder = async () => {
    setPicking(true);
    try {
      const res = await pickDatasetFolder();
      if (!res.cancelled && res.path) setData({ base_dir: res.path });
    } finally {
      setPicking(false);
    }
  };

  const collides = ignoreIndexCollides(formData);

  const card: React.CSSProperties = {
    background: "var(--vf-panel)",
    border: "1px solid var(--vf-panel-stroke)",
    borderRadius: 18,
    padding: 26,
    backdropFilter: "blur(14px)",
  };

  const warnBanner: React.CSSProperties = {
    padding: "12px 16px",
    background: "oklch(0.704 0.191 22.216 / 0.10)",
    border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
    borderRadius: 12,
    fontFamily: "var(--font-mono)",
    fontSize: 12,
    color: "oklch(0.85 0.14 22)",
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      {validationErrors.length > 0 && (
        <div style={warnBanner}>
          {validationErrors.slice(0, 5).map((e, i) => (
            <div key={i}>
              {e.field.join(" › ")}: {e.message}
            </div>
          ))}
        </div>
      )}

      {collides && (
        <div style={warnBanner}>
          ignore_index ({formData.data.ignore_index}) colide com um id de classe
          real (0…{formData.model.num_classes - 1}). Use um valor fora desse
          intervalo (ex. 255 ou -1).
        </div>
      )}

      {/* Experimento */}
      <div style={card}>
        <div style={sectionLabel}>Experimento</div>
        <div style={{ maxWidth: 420 }}>
          <TextField
            label="Nome do experimento"
            value={formData.name}
            onChange={(v) => setFormData((p) => ({ ...p, name: v }))}
            placeholder="segmentation_001"
            hint="usado na pasta de saída e no histórico"
            mono
          />
        </div>
      </div>

      {/* Modelo */}
      <div style={card}>
        <div style={sectionLabel}>Modelo · segmentação</div>
        <div style={grid}>
          <SelectField
            label="Arquitetura"
            value={formData.model.name}
            onChange={(v) => setModel({ name: v })}
            options={SEGMENTATION_MODELS}
            hint="dense head"
          />
          <NumberField
            label="Nº de classes"
            value={formData.model.num_classes}
            onChange={(v) => setModel({ num_classes: Math.round(v) })}
            min={1}
            step={1}
            hint="inclui fundo"
          />
          <Toggle
            label="Pesos pré-treinados"
            value={formData.model.pretrained}
            onChange={(v) => setModel({ pretrained: v })}
            hint="backbone ImageNet"
          />
        </div>
      </div>

      {/* Dataset */}
      <div style={card}>
        <div style={sectionLabel}>Dataset (imagens + máscaras)</div>
        <div style={grid}>
          <div
            style={{
              gridColumn: "1 / -1",
              display: "flex",
              gap: 10,
              alignItems: "flex-end",
            }}
          >
            <div style={{ flex: 1 }}>
              <TextField
                label="Pasta base"
                value={formData.data.base_dir}
                onChange={(v) => setData({ base_dir: v })}
                placeholder="…/dataset (train/{images,masks}, val/…)"
                hint="raiz do dataset"
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
          <TextField
            label="Subpasta de imagens"
            value={formData.data.images_subdir}
            onChange={(v) => setData({ images_subdir: v })}
            hint="por split"
            mono
          />
          <TextField
            label="Subpasta de máscaras"
            value={formData.data.masks_subdir}
            onChange={(v) => setData({ masks_subdir: v })}
            hint="PNG · id por pixel"
            mono
          />
          <TextField
            label="Split de treino"
            value={formData.data.train_dir}
            onChange={(v) => setData({ train_dir: v })}
            mono
          />
          <TextField
            label="Split de validação"
            value={formData.data.val_dir}
            onChange={(v) => setData({ val_dir: v })}
            mono
          />
          <TextField
            label="Split de teste"
            value={formData.data.test_dir}
            onChange={(v) => setData({ test_dir: v })}
            hint="opcional"
            mono
          />
          <NumberField
            label="Image size"
            value={formData.data.image_size}
            onChange={(v) => setData({ image_size: Math.round(v) })}
            min={32}
            step={32}
            suffix="px"
          />
          <NumberField
            label="ignore_index"
            value={formData.data.ignore_index}
            onChange={(v) => setData({ ignore_index: Math.round(v) })}
            step={1}
            hint="pixels void"
          />
        </div>
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
            step={0.0001}
          />
          <Segmented
            label="Loss"
            value={formData.training.loss}
            onChange={(v) => setTraining({ loss: v })}
            options={SEGMENTATION_LOSSES}
            hint="critério por pixel"
          />
          <Segmented
            label="Otimizador"
            value={formData.training.optimizer}
            onChange={(v) => setTraining({ optimizer: v })}
            options={[
              { value: "adam", label: "Adam" },
              { value: "sgd", label: "SGD" },
              { value: "adamw", label: "AdamW" },
            ]}
          />
          <NumberField
            label="Early stop"
            value={formData.training.early_stopping_patience}
            onChange={(v) => setTraining({ early_stopping_patience: Math.round(v) })}
            min={1}
            step={1}
            hint="paciência"
          />
          <NumberField
            label="Seed"
            value={formData.training.seed}
            onChange={(v) => setTraining({ seed: Math.round(v) })}
            min={0}
            step={1}
          />
        </div>
      </div>

      {/* Transfer learning */}
      <div style={card}>
        <div style={sectionLabel}>Transfer learning</div>
        <div style={grid}>
          <Segmented
            label="Modo"
            value={formData.transfer}
            onChange={(v) =>
              setFormData((p) => ({
                ...p,
                transfer: v as SegmentationForm["transfer"],
              }))
            }
            options={[
              { value: "none", label: "Completo" },
              { value: "feature_extraction", label: "Feature extr." },
              { value: "fine_tuning", label: "Fine-tuning" },
            ]}
            hint="backbone pré-treinado (torchvision)"
          />
          {formData.transfer === "fine_tuning" && (
            <NumberField
              label="Backbone LR ×"
              value={formData.backbone_lr_multiplier}
              onChange={(v) =>
                setFormData((p) => ({ ...p, backbone_lr_multiplier: v }))
              }
              min={0.0001}
              max={1}
              step={0.05}
              hint="LR do backbone = LR × isto"
            />
          )}
        </div>
      </div>

      {onCompare && (
        <ComparisonCard
          modelOptions={SEGMENTATION_MODELS}
          metrics={COMPARE_METRICS}
          accent={accent}
          disabled={busy}
          onCompare={onCompare}
        />
      )}
      {onSweep && (
        <SweepCard
          metrics={COMPARE_METRICS}
          pathHints={SWEEP_PATH_HINTS}
          accent={accent}
          disabled={busy}
          onSweep={onSweep}
        />
      )}
    </div>
  );
}
