import { useEffect, useState } from "react";
import { fetchModelDefaults, type ModelDefaults } from "../api/client";

/** Says when the chosen architecture and optimizer were measured to fail.
 *
 * ADR-099 found that VGG, AlexNet, Swin and ConvNeXt predict a single class for
 * every image at the previous default of 1e-3 — an accuracy of 0.25 on four
 * classes, or exactly 0.50 on two, reported without comment. ADR-100 measured
 * the rate that trains each family instead.
 *
 * It suggests rather than applies. The whole failure being addressed is a
 * number arriving without the researcher knowing where it came from, and
 * silently rewriting their learning rate would be the same mistake wearing a
 * friendlier face. The button is the consent.
 */
export function ModelAdvice({
  architecture,
  optimizer,
  learningRate,
  baseDir,
  pretrained = true,
  onApply,
}: {
  architecture: string;
  optimizer: string;
  learningRate: number;
  baseDir?: string;
  pretrained?: boolean;
  onApply: (next: { optimizer: string; learning_rate: number }) => void;
}) {
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

  // Only speak when the current settings differ from what was measured.
  const rateDiffers = Math.abs(advice.learning_rate - learningRate) > 1e-12;
  const optimizerDiffers = advice.optimizer !== optimizer;
  if (!rateDiffers && !optimizerDiffers) return null;

  const severe = advice.collapse_prone;

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
      <div style={{ marginBottom: 8 }}>
        {advice.note ??
          `Para ${architecture}, o valor medido é ${advice.optimizer} a ${advice.learning_rate}.`}
      </div>
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
        usar {advice.optimizer} · {advice.learning_rate}
      </button>
    </div>
  );
}
