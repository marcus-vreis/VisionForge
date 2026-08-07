import { useEffect, useRef, useState } from "react";
import {
  fetchDetectionDatasetSamples,
  fetchDetectionDatasetStats,
  type DetectionDatasetSamplesResponse,
  type DetectionDatasetStatsResponse,
  type DetectionSplitStats,
} from "../api/client";

interface DetectionDatasetStatsProps {
  baseDir: string;
  /** Applies the detected class count to the detection form (once per base_dir). */
  onApplyClasses?: (numClasses: number, classNames: string[]) => void;
}

const SPLIT_LABELS: Record<string, string> = {
  train: "treino",
  val: "validação",
  test: "teste",
};

/** Pre-training overview for a YOLO dataset: image/instance counts per split,
 * per-class annotation distribution, unlabeled images, and imbalance flag.
 * The detection analogue of `DatasetStats` (which assumes ImageFolder). */
export function DetectionDatasetStats({ baseDir, onApplyClasses }: DetectionDatasetStatsProps) {
  const [stats, setStats] = useState<DetectionDatasetStatsResponse | null>(null);
  const [samples, setSamples] = useState<DetectionDatasetSamplesResponse | null>(
    null,
  );
  const appliedForBaseDir = useRef<string | null>(null);

  // All setState happens in the async callbacks (not synchronously in the effect
  // body) so the work stays a genuine external-system sync, not a cascading
  // render. Empty base_dir simply skips the fetch; the render guard hides us.
  // Crops are a separate request so a slow decode never delays the counts,
  // and a failure costs the thumbnails rather than the whole panel.
  useEffect(() => {
    if (!baseDir.trim()) return;
    let alive = true;
    fetchDetectionDatasetSamples(baseDir)
      .then((r) => {
        if (alive) setSamples(r);
      })
      .catch(() => {
        if (alive) setSamples(null);
      });
    return () => {
      alive = false;
    };
  }, [baseDir]);

  useEffect(() => {
    if (!baseDir.trim()) return;
    let alive = true;
    fetchDetectionDatasetStats(baseDir)
      .then((s) => {
        if (!alive) return;
        setStats(s);
        if (
          onApplyClasses &&
          s.class_names.length > 0 &&
          appliedForBaseDir.current !== baseDir
        ) {
          appliedForBaseDir.current = baseDir;
          onApplyClasses(s.class_names.length, s.class_names);
        }
      })
      .catch(() => {
        if (alive) setStats(null);
      });
    return () => {
      alive = false;
    };
  }, [baseDir]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!baseDir.trim() || !stats) return null;

  if (stats.class_names.length === 0) {
    return (
      <div
        style={{
          marginTop: 12,
          padding: "10px 14px",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          color: "oklch(0.85 0.14 22)",
          background: "oklch(0.704 0.191 22.216 / 0.10)",
          border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
          borderRadius: 10,
        }}
      >
        {stats.message ?? "Nenhum split YOLO encontrado (images/<split>)."}
      </div>
    );
  }

  return (
    <div
      style={{
        marginTop: 12,
        padding: "12px 16px",
        background: "rgba(255,255,255,0.025)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 12,
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      {/* Class id → name map (YOLO ids are explicit in the label files) */}
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8 }}>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 9,
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            color: "var(--vf-text-muted)",
          }}
        >
          mapeamento (YOLO):
        </span>
        {stats.class_names.map((cn, idx) => (
          <span
            key={cn}
            style={{
              padding: "3px 9px",
              background: "rgba(255,255,255,0.04)",
              border: "1px solid var(--vf-panel-stroke)",
              borderRadius: 999,
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "var(--vf-text-dim)",
            }}
          >
            <span style={{ color: "var(--accent-vf)", marginRight: 6 }}>{idx}</span>
            {cn}
          </span>
        ))}
        {onApplyClasses && stats.class_names.length > 0 && (
          <span
            title={`Detectado e aplicado ao config: num_classes=${stats.class_names.length}`}
            style={{
              marginLeft: "auto",
              padding: "3px 9px",
              background: "oklch(0.72 0.16 150 / 0.12)",
              border: "1px solid oklch(0.72 0.16 150 / 0.4)",
              borderRadius: 999,
              fontFamily: "var(--font-mono)",
              fontSize: 9,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              color: "oklch(0.85 0.16 150)",
            }}
          >
            ✓ {stats.class_names.length} classe(s) aplicada(s)
          </span>
        )}
      </div>

      {samples && Object.keys(samples.crops).length > 0 && (
        <div>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              color: "var(--vf-text-muted)",
              marginBottom: 10,
            }}
          >
            // amostras (split: {samples.split}) — sanity-check de labels
          </div>
          {Object.entries(samples.crops).map(([cn, uris]) => (
            <div
              key={cn}
              style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}
            >
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  color: "var(--vf-text-dim)",
                  minWidth: 110,
                }}
              >
                {cn}
              </span>
              {uris.map((uri, i) => (
                <img
                  key={i}
                  src={uri}
                  alt={`${cn} — exemplo ${i + 1}`}
                  style={{
                    height: 56,
                    borderRadius: 6,
                    border: "1px solid var(--vf-panel-stroke)",
                  }}
                />
              ))}
            </div>
          ))}
        </div>
      )}

      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "var(--vf-text-muted)",
          }}
        >
          // distribuição de anotações (instâncias)
        </div>
        {stats.imbalanced && (
          <span
            style={{
              padding: "3px 8px",
              fontFamily: "var(--font-mono)",
              fontSize: 9,
              borderRadius: 6,
              background: "oklch(0.78 0.16 75 / 0.16)",
              color: "oklch(0.85 0.14 75)",
              border: "1px solid oklch(0.78 0.16 75 / 0.5)",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
            }}
          >
            ⚠ desbalanceado
          </span>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
        {(["train", "val", "test"] as const).map((splitKey) => {
          const split = stats.splits[splitKey];
          if (!split) return null;
          return (
            <DetectionSplitCard
              key={splitKey}
              label={SPLIT_LABELS[splitKey]}
              split={split}
              classNames={stats.class_names}
            />
          );
        })}
      </div>
    </div>
  );
}

interface DetectionSplitCardProps {
  label: string;
  split: DetectionSplitStats;
  classNames: string[];
}

function DetectionSplitCard({ label, split, classNames }: DetectionSplitCardProps) {
  if (split.missing) {
    return (
      <div
        style={{
          padding: "10px 12px",
          background: "rgba(0,0,0,0.25)",
          border: "1px dashed var(--vf-panel-stroke)",
          borderRadius: 8,
          opacity: 0.55,
        }}
      >
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 9,
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            color: "var(--vf-text-muted)",
          }}
        >
          {label}
        </div>
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            marginTop: 6,
            color: "var(--vf-text-muted)",
          }}
        >
          ausente
        </div>
      </div>
    );
  }

  const totalAnn = split.total_annotations;
  return (
    <div
      style={{
        padding: "10px 12px",
        background: "rgba(0,0,0,0.30)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 8,
        display: "flex",
        flexDirection: "column",
        gap: 6,
      }}
    >
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 9,
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            color: "var(--vf-text-muted)",
          }}
        >
          {label}
        </span>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 13,
            fontWeight: 600,
            color: "var(--vf-text)",
          }}
        >
          {split.total_images} img
        </span>
      </div>
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 9,
          color: "var(--vf-text-muted)",
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <span>{totalAnn} caixas</span>
        {split.unlabeled_images > 0 && (
          <span style={{ color: "oklch(0.85 0.14 75)" }}>
            {split.unlabeled_images} sem label
          </span>
        )}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        {classNames.map((cn) => {
          const count = split.class_counts[cn] ?? 0;
          const pct = totalAnn === 0 ? 0 : (count / totalAnn) * 100;
          return (
            <div key={cn} style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontFamily: "var(--font-mono)",
                  fontSize: 9,
                  color: "var(--vf-text-dim)",
                }}
              >
                <span style={{ wordBreak: "break-all" }}>{cn}</span>
                <span>{count}</span>
              </div>
              <div
                style={{
                  height: 4,
                  width: "100%",
                  background: "rgba(255,255,255,0.05)",
                  borderRadius: 2,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${pct}%`,
                    background: "var(--accent-vf)",
                    transition: "width 200ms ease",
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
