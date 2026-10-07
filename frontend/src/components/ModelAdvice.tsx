import { useEffect, useState } from "react";
import { fetchModelDefaults, type ModelDefaults } from "../api/client";
import { useT } from "../i18n/useT";
import { isAlarming, modelAdviceNote, type ModelAdviceTask } from "../lib/model-advice";

/** Says when the chosen architecture and optimizer were measured to fail.
 *
 * ADR-099 found that vgg16 and alexnet predict a single class for every image
 * at the previous default of 1e-3 — an accuracy of 0.25 on four classes, or
 * exactly 0.50 on two, reported without comment. ADR-100 found the same for
 * swin_t and convnext_tiny, while vit_b_16 learned little (0.41), and
 * measured the rate that trains each. The note quotes only that: one model per
 * family, named, and no number at all for a family that was never run. The alarm
 * colour goes with the measurement: a family flagged without one (maxvit) still
 * gets the suggestion, in the plain style of the other notes. The numbers were
 * measured on classification, so the regression and segmentation forms pass
 * their `task` and the note labels them as classification's.
 *
 * It suggests rather than applies. The whole failure being addressed is a
 * number arriving without the researcher knowing where it came from, and
 * silently rewriting their learning rate would be the same mistake wearing a
 * friendlier face. The button is the consent.
 */
export function ModelAdvice({
  task,
  architecture,
  optimizer,
  learningRate,
  baseDir,
  pretrained = true,
  onApply,
}: {
  task: ModelAdviceTask;
  architecture: string;
  optimizer: string;
  learningRate: number;
  baseDir?: string;
  pretrained?: boolean;
  onApply: (next: { optimizer: string; learning_rate: number }) => void;
}) {
  const t = useT();
  const [advice, setAdvice] = useState<ModelDefaults | null>(null);

  useEffect(() => {
    if (!architecture) return;
    let alive = true;
    fetchModelDefaults(architecture, baseDir, pretrained)
      .then((d) => alive && setAdvice(d))
      .catch(() => alive && setAdvice(null));
    return () => {
      alive = false;
    };
  }, [architecture, baseDir, pretrained]);

  if (!advice) return null;

  // Only speak when the current settings differ from the suggestion.
  const rateDiffers = Math.abs(advice.learning_rate - learningRate) > 1e-12;
  const optimizerDiffers = advice.optimizer !== optimizer;
  if (!rateDiffers && !optimizerDiffers) return null;

  const severe = isAlarming(advice, task);

  return (
    <div
      style={{
        marginTop: 10,
        padding: "10px 12px",
        borderRadius: 8,
        border: `1px solid ${severe ? "oklch(0.80 0.16 85 / 0.45)" : "var(--vf-panel-stroke)"}`,
        background: severe
          ? "oklch(0.80 0.16 85 / 0.10)"
          : "rgba(255,255,255,0.03)",
        fontSize: 11,
        lineHeight: 1.5,
        color: "var(--vf-text-dim)",
      }}
    >
      <div style={{ marginBottom: 8 }}>{modelAdviceNote(t, advice, task)}</div>
      <button
        type="button"
        onClick={() =>
          onApply({
            optimizer: advice.optimizer,
            learning_rate: advice.learning_rate,
          })
        }
        style={{
          padding: "5px 10px",
          borderRadius: 6,
          border: "1px solid var(--accent-vf)",
          background: "var(--accent-soft)",
          color: "var(--vf-text)",
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          cursor: "pointer",
        }}
      >
        {t.modelAdvice.apply(advice.optimizer, advice.learning_rate)}
      </button>
    </div>
  );
}
