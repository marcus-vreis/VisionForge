import { useState } from "react";
import { pickDatasetFolder, pickDetectionYaml } from "../api/client";
import {
  DETECTION_BACKENDS,
  DETECTION_MODELS,
  DETECTION_OPTIMIZERS,
  defaultModelForBackend,
  isValidModelForBackend,
  type DetectionAugmentationForm,
  type DetectionBackend,
  type DetectionForm,
  type DetectionOptimizer,
} from "../lib/detection-models";
import type { ValidationError } from "../hooks/useExperiment";
import { NumberField, SelectField, Segmented, TextField, Toggle } from "./controls";
import { DetectionDatasetStats } from "./DetectionDatasetStats";
import { ComparisonCard } from "./ComparisonCard";
import { SweepCard, type SweepPayload } from "./SweepCard";

const COMPARE_METRICS = [
  { value: "map50_95", label: "mAP@50-95" },
  { value: "map50", label: "mAP@50" },
];

const SWEEP_PATH_HINTS = [
  "training.learning_rate",
  "training.epochs",
  "model.name",
];

interface DetectionPanelProps {
  formData: DetectionForm;
  setFormData: (updater: (prev: DetectionForm) => DetectionForm) => void;
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

/** Schema-aligned form for an Ultralytics/torchvision detection run. */
export function DetectionPanel({
  formData,
  setFormData,
  accent,
  validationErrors,
  busy,
  onCompare,
  onSweep,
}: DetectionPanelProps) {
  const [picking, setPicking] = useState(false);

  const setModel = (patch: Partial<DetectionForm["model"]>) =>
    setFormData((p) => ({ ...p, model: { ...p.model, ...patch } }));
  const setData = (patch: Partial<DetectionForm["data"]>) =>
    setFormData((p) => ({ ...p, data: { ...p.data, ...patch } }));
  const setTraining = (patch: Partial<DetectionForm["training"]>) =>
    setFormData((p) => ({ ...p, training: { ...p.training, ...patch } }));
  const setAug = (patch: Partial<DetectionAugmentationForm>) =>
    setFormData((p) => ({
      ...p,
      training: {
        ...p.training,
        augmentation: { ...p.training.augmentation, ...patch },
      },
    }));

  const isUltralytics = formData.model.backend === "ultralytics";

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

  const onPickYaml = async () => {
    setPicking(true);
    try {
      const res = await pickDetectionYaml();
      if (!res.cancelled && res.path) setData({ data_yaml: res.path });
    } finally {
      setPicking(false);
    }
  };

  const pickButton: React.CSSProperties = {
    padding: "12px 16px",
    background: "var(--accent-soft)",
    border: `1px solid ${accent}`,
    borderRadius: 10,
    color: "var(--vf-text)",
    fontFamily: "var(--font-mono)",
    fontSize: 12,
    cursor: picking ? "default" : "pointer",
    whiteSpace: "nowrap",
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

      {/* Experimento */}
      <div style={card}>
        <div style={sectionLabel}>Experimento</div>
        <div style={{ maxWidth: 420 }}>
          <TextField
            label="Nome do experimento"
            value={formData.name}
            onChange={(v) => setFormData((p) => ({ ...p, name: v }))}
            placeholder="detection_001"
            hint="usado na pasta de saída e no histórico"
            mono
          />
        </div>
      </div>

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
        <div style={sectionLabel}>Dataset</div>
        <div style={{ marginBottom: 14, maxWidth: 360 }}>
          <Segmented
            label="Fonte do dataset"
            value={formData.data.source}
            onChange={(v) =>
              setData({ source: v as DetectionForm["data"]["source"] })
            }
            options={[
              { value: "folder", label: "Pasta YOLO" },
              { value: "yaml", label: "data.yaml" },
            ]}
            hint={
              formData.data.source === "folder"
                ? "gera o data.yaml a partir da pasta"
                : "usa um data.yaml existente"
            }
          />
        </div>
        <div style={grid}>
          {formData.data.source === "folder" ? (
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
                style={pickButton}
              >
                {picking ? "…" : "📁 Escolher"}
              </button>
            </div>
          ) : (
            <div style={{ gridColumn: "1 / -1", display: "flex", gap: 10, alignItems: "flex-end" }}>
              <div style={{ flex: 1 }}>
                <TextField
                  label="Arquivo data.yaml"
                  value={formData.data.data_yaml}
                  onChange={(v) => setData({ data_yaml: v })}
                  placeholder="…/dataset/data.yaml"
                  hint="Ultralytics / Roboflow"
                  mono
                />
              </div>
              <button
                type="button"
                onClick={() => void onPickYaml()}
                disabled={picking}
                style={pickButton}
              >
                {picking ? "…" : "📄 Escolher"}
              </button>
            </div>
          )}
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
        {formData.data.source === "folder" ? (
          <DetectionDatasetStats
            baseDir={formData.data.base_dir}
            onApplyClasses={(n) => setModel({ num_classes: n })}
          />
        ) : (
          <p
            style={{
              marginTop: 14,
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              lineHeight: 1.6,
              color: "var(--vf-text-muted)",
            }}
          >
            O data.yaml define splits e nomes de classe. Confirme que{" "}
            <code>nc</code> bate com o nº de classes acima.
          </p>
        )}
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
          <SelectField
            label="Optimizer"
            value={formData.training.optimizer}
            onChange={(v) =>
              setTraining({ optimizer: v as DetectionOptimizer })
            }
            options={DETECTION_OPTIMIZERS.map((o) => ({ value: o, label: o }))}
            hint="auto = Ultralytics escolhe"
          />
          <NumberField
            label="Momentum"
            value={formData.training.momentum}
            onChange={(v) => setTraining({ momentum: v })}
            min={0}
            max={1}
            step={0.001}
            hint="SGD momentum / Adam β1"
          />
          <NumberField
            label="Weight decay"
            value={formData.training.weight_decay}
            onChange={(v) => setTraining({ weight_decay: v })}
            min={0}
            step={0.0001}
            hint="L2"
          />
        </div>
      </div>

      {isUltralytics && (
        <>
          {/* Schedule & loss */}
          <div style={card}>
            <div style={sectionLabel}>Schedule de LR & pesos de loss</div>
            <div style={grid}>
              <NumberField
                label="lrf"
                value={formData.training.lrf}
                onChange={(v) => setTraining({ lrf: v })}
                min={0.000001}
                step={0.001}
                hint="LR final = lr0 × lrf"
              />
              <Toggle
                label="Cosine LR"
                value={formData.training.cos_lr}
                onChange={(v) => setTraining({ cos_lr: v })}
                hint="schedule cosseno"
              />
              <NumberField
                label="Warmup epochs"
                value={formData.training.warmup_epochs}
                onChange={(v) => setTraining({ warmup_epochs: v })}
                min={0}
                step={0.5}
              />
              <NumberField
                label="Warmup momentum"
                value={formData.training.warmup_momentum}
                onChange={(v) => setTraining({ warmup_momentum: v })}
                min={0}
                max={1}
                step={0.05}
              />
              <NumberField
                label="Warmup bias LR"
                value={formData.training.warmup_bias_lr}
                onChange={(v) => setTraining({ warmup_bias_lr: v })}
                min={0}
                step={0.01}
              />
              <NumberField
                label="Box loss gain"
                value={formData.training.box}
                onChange={(v) => setTraining({ box: v })}
                min={0}
                step={0.1}
                hint="box"
              />
              <NumberField
                label="Cls loss gain"
                value={formData.training.cls}
                onChange={(v) => setTraining({ cls: v })}
                min={0}
                step={0.1}
                hint="cls"
              />
              <NumberField
                label="DFL loss gain"
                value={formData.training.dfl}
                onChange={(v) => setTraining({ dfl: v })}
                min={0}
                step={0.1}
                hint="dfl"
              />
            </div>
          </div>

          {/* Regularization & mechanics */}
          <div style={card}>
            <div style={sectionLabel}>Regularização & mecânica</div>
            <div style={grid}>
              <NumberField
                label="Label smoothing"
                value={formData.training.label_smoothing}
                onChange={(v) => setTraining({ label_smoothing: v })}
                min={0}
                max={1}
                step={0.01}
              />
              <NumberField
                label="Dropout"
                value={formData.training.dropout}
                onChange={(v) => setTraining({ dropout: v })}
                min={0}
                max={1}
                step={0.05}
              />
              <NumberField
                label="Nominal batch (nbs)"
                value={formData.training.nbs}
                onChange={(v) => setTraining({ nbs: Math.round(v) })}
                min={1}
                step={1}
              />
              <NumberField
                label="Freeze layers"
                value={formData.training.freeze}
                onChange={(v) => setTraining({ freeze: Math.round(v) })}
                min={0}
                step={1}
                hint="0 = nenhuma"
              />
              <NumberField
                label="Close mosaic"
                value={formData.training.close_mosaic}
                onChange={(v) => setTraining({ close_mosaic: Math.round(v) })}
                min={0}
                step={1}
                hint="desliga mosaico nas últimas N épocas"
              />
              <Toggle
                label="AMP"
                value={formData.training.amp}
                onChange={(v) => setTraining({ amp: v })}
                hint="precisão mista"
              />
              <Toggle
                label="Single class"
                value={formData.training.single_cls}
                onChange={(v) => setTraining({ single_cls: v })}
                hint="trata tudo como 1 classe"
              />
              <Toggle
                label="Rect"
                value={formData.training.rect}
                onChange={(v) => setTraining({ rect: v })}
                hint="batches retangulares"
              />
              <Toggle
                label="Multi-scale"
                value={formData.training.multi_scale}
                onChange={(v) => setTraining({ multi_scale: v })}
                hint="varia imgsz ±50%"
              />
            </div>
          </div>

          {/* Augmentation */}
          <div style={card}>
            <div style={sectionLabel}>Data augmentation</div>
            <div style={grid}>
              <NumberField
                label="HSV — hue"
                value={formData.training.augmentation.hsv_h}
                onChange={(v) => setAug({ hsv_h: v })}
                min={0}
                max={1}
                step={0.005}
              />
              <NumberField
                label="HSV — saturation"
                value={formData.training.augmentation.hsv_s}
                onChange={(v) => setAug({ hsv_s: v })}
                min={0}
                max={1}
                step={0.05}
              />
              <NumberField
                label="HSV — value"
                value={formData.training.augmentation.hsv_v}
                onChange={(v) => setAug({ hsv_v: v })}
                min={0}
                max={1}
                step={0.05}
              />
              <NumberField
                label="Degrees"
                value={formData.training.augmentation.degrees}
                onChange={(v) => setAug({ degrees: v })}
                min={-180}
                max={180}
                step={1}
                suffix="°"
              />
              <NumberField
                label="Translate"
                value={formData.training.augmentation.translate}
                onChange={(v) => setAug({ translate: v })}
                min={0}
                max={1}
                step={0.05}
              />
              <NumberField
                label="Scale"
                value={formData.training.augmentation.scale}
                onChange={(v) => setAug({ scale: v })}
                min={0}
                step={0.05}
              />
              <NumberField
                label="Shear"
                value={formData.training.augmentation.shear}
                onChange={(v) => setAug({ shear: v })}
                min={-180}
                max={180}
                step={1}
                suffix="°"
              />
              <NumberField
                label="Perspective"
                value={formData.training.augmentation.perspective}
                onChange={(v) => setAug({ perspective: v })}
                min={0}
                max={0.001}
                step={0.0001}
              />
              <NumberField
                label="Flip up-down"
                value={formData.training.augmentation.flipud}
                onChange={(v) => setAug({ flipud: v })}
                min={0}
                max={1}
                step={0.05}
                hint="probabilidade"
              />
              <NumberField
                label="Flip left-right"
                value={formData.training.augmentation.fliplr}
                onChange={(v) => setAug({ fliplr: v })}
                min={0}
                max={1}
                step={0.05}
                hint="probabilidade"
              />
              <NumberField
                label="BGR swap"
                value={formData.training.augmentation.bgr}
                onChange={(v) => setAug({ bgr: v })}
                min={0}
                max={1}
                step={0.05}
                hint="probabilidade"
              />
              <NumberField
                label="Mosaic"
                value={formData.training.augmentation.mosaic}
                onChange={(v) => setAug({ mosaic: v })}
                min={0}
                max={1}
                step={0.05}
                hint="probabilidade"
              />
              <NumberField
                label="Mixup"
                value={formData.training.augmentation.mixup}
                onChange={(v) => setAug({ mixup: v })}
                min={0}
                max={1}
                step={0.05}
                hint="probabilidade"
              />
              <NumberField
                label="Copy-paste"
                value={formData.training.augmentation.copy_paste}
                onChange={(v) => setAug({ copy_paste: v })}
                min={0}
                max={1}
                step={0.05}
                hint="probabilidade"
              />
              <SelectField
                label="Auto augment"
                value={formData.training.augmentation.auto_augment}
                onChange={(v) =>
                  setAug({
                    auto_augment:
                      v as DetectionAugmentationForm["auto_augment"],
                  })
                }
                options={[
                  { value: "randaugment", label: "RandAugment" },
                  { value: "autoaugment", label: "AutoAugment" },
                  { value: "augmix", label: "AugMix" },
                  { value: "none", label: "Desativado" },
                ]}
              />
              <NumberField
                label="Random erasing"
                value={formData.training.augmentation.erasing}
                onChange={(v) => setAug({ erasing: v })}
                min={0}
                max={1}
                step={0.05}
                hint="probabilidade"
              />
            </div>
          </div>
        </>
      )}

      {onCompare && (
        <ComparisonCard
          modelOptions={DETECTION_MODELS[formData.model.backend]}
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
