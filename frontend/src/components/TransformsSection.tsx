import { AugmentPreview } from "./AugmentPreview";
import {
  PreprocessingPanel,
  type PreprocessingStep,
} from "./PreprocessingPanel";
import { NumberField, TextField, Toggle } from "./controls";
import {
  buildTransformsPayload,
  type TransformsForm,
} from "../lib/transforms-form";

interface TransformsSectionProps {
  baseDir: string;
  steps: PreprocessingStep[];
  onStepsChange: (steps: PreprocessingStep[]) => void;
  transforms: TransformsForm;
  onTransformsChange: (patch: Partial<TransformsForm>) => void;
  /** The task's own image_size, forwarded so the augment preview renders at
   *  the size training will actually use. */
  imageSize?: number;
}

const sectionLabel: React.CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  letterSpacing: "0.22em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
  marginBottom: 12,
};

const card: React.CSSProperties = {
  background: "var(--vf-panel)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 18,
  padding: 26,
  backdropFilter: "blur(14px)",
};

const grid: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
  gap: 14,
};

/** Preprocessing pipeline + augmentation/normalization for the standalone task
 *  panels (ADR-059) — the canonical sections 6 and 7 of the task-panel
 *  contract. The values mirror `data.preprocessing` / `data.transforms`, which
 *  the regression/segmentation/anomaly backends already consume; before this
 *  section existed those defaults applied invisibly. */
export function TransformsSection({
  baseDir,
  steps,
  onStepsChange,
  transforms,
  onTransformsChange,
  imageSize,
}: TransformsSectionProps) {
  return (
    <>
      <div style={card}>
        <PreprocessingPanel
          baseDir={baseDir}
          steps={steps}
          onChange={onStepsChange}
        />
      </div>

      {/* Normalization is not augmentation: it applies to train, val and test
          alike, so it stays visible when augmentation is switched off. The old
          combined section said otherwise by putting them under one heading. */}
      <div style={card}>
        <div style={sectionLabel}>Imagem</div>
        <div style={grid}>
          <TextField
            label="Normalização (média)"
            value={transforms.normalize_mean}
            onChange={(v) => onTransformsChange({ normalize_mean: v })}
            hint="R, G, B — aplicada a treino, validação e teste"
          />
          <TextField
            label="Normalização (std)"
            value={transforms.normalize_std}
            onChange={(v) => onTransformsChange({ normalize_std: v })}
            hint="R, G, B — aplicada a treino, validação e teste"
          />
        </div>
      </div>

      <div style={card}>
        <div style={sectionLabel}>Data augmentation</div>
        <Toggle
          label="Augmentation"
          value={transforms.augment}
          onChange={(v) => onTransformsChange({ augment: v })}
          hint="só no treino; desligada, os valores abaixo ficam guardados"
        />
        {transforms.augment ? (
          <>
            <div style={{ ...grid, marginTop: 14 }}>
              <Toggle
                label="Flip horizontal"
                value={transforms.horizontal_flip}
                onChange={(v) => onTransformsChange({ horizontal_flip: v })}
                hint="treino"
              />
              <NumberField
                label="Rotação (graus)"
                value={transforms.rotation_degrees}
                onChange={(v) =>
                  onTransformsChange({ rotation_degrees: Math.max(Math.round(v), 0) })
                }
                min={0}
                step={1}
                hint="0 = desliga"
              />
              <Toggle
                label="Color jitter"
                value={transforms.color_jitter}
                onChange={(v) => onTransformsChange({ color_jitter: v })}
              />
            </div>
            {/* Previewing transforms the run will not apply would show something
                that never happens, so the preview follows the toggle. */}
            <div style={{ marginTop: 16 }}>
              <AugmentPreview
                baseDir={baseDir}
                transforms={buildTransformsPayload(transforms, imageSize)}
              />
            </div>
          </>
        ) : (
          <div
            style={{
              marginTop: 14,
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              color: "var(--vf-text-muted)",
            }}
          >
            3 parâmetros ocultos — ligue para ajustar
          </div>
        )}
      </div>
    </>
  );
}
