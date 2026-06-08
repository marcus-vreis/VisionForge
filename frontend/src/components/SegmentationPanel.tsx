import { useState } from "react";
import { pickDatasetFolder } from "../api/client";
import {
  SEGMENTATION_COMPARE_METRICS,
  SEGMENTATION_LOSSES,
  SEGMENTATION_MODELS,
  ignoreIndexCollides,
  type SegmentationForm,
} from "../lib/segmentation-models";
import type { ValidationError } from "../hooks/useExperiment";
import { NumberField, SelectField, Segmented, TextField, Toggle } from "./controls";

interface SegmentationPanelProps {
  formData: SegmentationForm;
  setFormData: (updater: (prev: SegmentationForm) => SegmentationForm) => void;
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

/** Schema-aligned form for a paired image/mask semantic-segmentation run. */
export function SegmentationPanel({
  formData,
  setFormData,
  accent,
  validationErrors,
}: SegmentationPanelProps) {
  const [picking, setPicking] = useState(false);

  const setModel = (patch: Partial<SegmentationForm["model"]>) =>
    setFormData((p) => ({ ...p, model: { ...p.model, ...patch } }));
  const setData = (patch: Partial<SegmentationForm["data"]>) =>
    setFormData((p) => ({ ...p, data: { ...p.data, ...patch } }));
  const setTraining = (patch: Partial<SegmentationForm["training"]>) =>
    setFormData((p) => ({ ...p, training: { ...p.training, ...patch } }));
  const setCompare = (patch: Partial<SegmentationForm["compare"]>) =>
    setFormData((p) => ({ ...p, compare: { ...p.compare, ...patch } }));
  const toggleArch = (name: string) =>
    setFormData((p) => {
      const has = p.compare.model_names.includes(name);
      const model_names = has
        ? p.compare.model_names.filter((n) => n !== name)
        : [...p.compare.model_names, name];
      return { ...p, compare: { ...p.compare, model_names } };
    });

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

      {/* Comparar modelos */}
      <div style={card}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 16,
          }}
        >
          <div style={{ ...sectionLabel, marginBottom: 0 }}>Comparar modelos</div>
          <div style={{ minWidth: 150 }}>
            <Toggle
              label="Ativar"
              value={formData.compare.enabled}
              onChange={(v) => setCompare({ enabled: v })}
              hint="ranquear arquiteturas"
            />
          </div>
        </div>
        {formData.compare.enabled && (
          <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 14 }}>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--vf-text-muted)",
                lineHeight: 1.5,
              }}
            >
              Treina cada arquitetura selecionada no mesmo dataset e ranqueia
              pela métrica. A arquitetura única acima é ignorada neste modo.
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {SEGMENTATION_MODELS.map((m) => {
                const selected = formData.compare.model_names.includes(m.value);
                return (
                  <button
                    key={m.value}
                    type="button"
                    aria-pressed={selected}
                    onClick={() => toggleArch(m.value)}
                    style={{
                      padding: "7px 12px",
                      borderRadius: 999,
                      cursor: "pointer",
                      fontFamily: "var(--font-mono)",
                      fontSize: 12,
                      color: selected ? "var(--vf-text)" : "var(--vf-text-muted)",
                      background: selected ? "var(--accent-soft)" : "transparent",
                      border: `1px solid ${selected ? accent : "var(--vf-panel-stroke)"}`,
                    }}
                  >
                    {m.label}
                  </button>
                );
              })}
            </div>
            <div style={{ maxWidth: 220 }}>
              <SelectField
                label="Métrica de ranking"
                value={formData.compare.metric}
                onChange={(v) => setCompare({ metric: v })}
                options={SEGMENTATION_COMPARE_METRICS}
                hint="maior é melhor"
              />
            </div>
            {formData.compare.model_names.length < 2 && (
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  color: "oklch(0.85 0.14 22)",
                }}
              >
                Selecione ao menos 2 arquiteturas para comparar.
              </div>
            )}
          </div>
        )}
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
    </div>
  );
}
