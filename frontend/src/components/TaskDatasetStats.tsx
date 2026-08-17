import { useEffect, useState } from "react";
import {
  fetchAnomalyDatasetStats,
  fetchRegressionDatasetStats,
  fetchSegmentationDatasetStats,
  type AnomalyDatasetStatsResponse,
  type RegressionDatasetStatsResponse,
  type SegmentationDatasetStatsResponse,
} from "../api/client";

/** Pre-training dataset overviews for segmentation / anomaly / regression
 *  (ADR-059) — the task-specific analogues of `DetectionDatasetStats`,
 *  rendered inside each panel's Dataset section. Fetches are debounced and
 *  skipped while the base dir is empty. */

const SPLIT_LABELS: Record<string, string> = {
  train: "treino",
  val: "validação",
  test: "teste",
};

const box: React.CSSProperties = {
  marginTop: 12,
  padding: "12px 16px",
  background: "rgba(255,255,255,0.025)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 12,
  display: "flex",
  flexDirection: "column",
  gap: 10,
};

const warnBox: React.CSSProperties = {
  marginTop: 12,
  padding: "10px 14px",
  fontFamily: "var(--font-mono)",
  fontSize: 11,
  color: "oklch(0.85 0.14 22)",
  background: "oklch(0.704 0.191 22.216 / 0.10)",
  border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
  borderRadius: 10,
};

const kicker: React.CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  letterSpacing: "0.18em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
};

const splitCard: React.CSSProperties = {
  padding: "10px 12px",
  background: "rgba(0,0,0,0.30)",
  border: "1px solid var(--vf-panel-stroke)",
  borderRadius: 8,
  display: "flex",
  flexDirection: "column",
  gap: 6,
};

const splitLabel: React.CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 9,
  letterSpacing: "0.14em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
};

const bigNumber: React.CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 13,
  fontWeight: 600,
  color: "var(--vf-text)",
};

const smallLine: React.CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 9,
  color: "var(--vf-text-muted)",
};

const warnColor = "oklch(0.85 0.14 75)";

function MissingSplitCard({ label }: { label: string }) {
  return (
    <div style={{ ...splitCard, border: "1px dashed var(--vf-panel-stroke)", opacity: 0.55 }}>
      <div style={splitLabel}>{label}</div>
      <div style={{ ...smallLine, fontSize: 12, marginTop: 4 }}>ausente</div>
    </div>
  );
}

/** Debounce a fetch keyed on serialized params; null while empty/failed. */
function useDebouncedStats<T>(
  key: string,
  enabled: boolean,
  fetcher: () => Promise<T>,
): T | null {
  const [stats, setStats] = useState<T | null>(null);
  useEffect(() => {
    // Disabled means "no stats to show", which is a question about the current
    // render rather than a state change -- clearing it here would be a setState
    // in an effect body, and a cascading render for a value already derivable.
    if (!enabled) return;
    let alive = true;
    const timer = setTimeout(() => {
      fetcher()
        .then((s) => {
          if (alive) setStats(s);
        })
        .catch(() => {
          if (alive) setStats(null);
        });
    }, 400);
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [key, enabled]); // eslint-disable-line react-hooks/exhaustive-deps
  return enabled ? stats : null;
}

// ---------------------------------------------------------------- segmentation

interface SegmentationDatasetStatsProps {
  baseDir: string;
  imagesSubdir: string;
  masksSubdir: string;
  trainDir: string;
  valDir: string;
  testDir: string;
  /** Injects the suggested class count (from sampled mask ids) into the form. */
  onApplyClasses?: (numClasses: number) => void;
}

export function SegmentationDatasetStats(props: SegmentationDatasetStatsProps) {
  const { baseDir, onApplyClasses } = props;
  const body = {
    base_dir: baseDir,
    images_subdir: props.imagesSubdir,
    masks_subdir: props.masksSubdir,
    train_dir: props.trainDir,
    val_dir: props.valDir,
    test_dir: props.testDir,
  };
  const stats = useDebouncedStats<SegmentationDatasetStatsResponse>(
    JSON.stringify(body),
    baseDir.trim() !== "",
    () => fetchSegmentationDatasetStats(body),
  );
  if (!stats) return null;
  if (stats.message) return <div style={warnBox}>{stats.message}</div>;

  // ids ≥ 200 are almost certainly void/ignore markers (e.g. 255), not classes.
  const classIds = stats.mask_class_ids.filter((i) => i < 200);
  // Dozens of distinct ids means interpolated/anti-aliased masks, not class
  // maps — a classic segmentation-dataset bug. Warn instead of suggesting.
  const looksInterpolated = classIds.length > 32;
  const suggested =
    !looksInterpolated && classIds.length > 0 ? Math.max(...classIds) + 1 : 0;
  const shownIds = stats.mask_class_ids.slice(0, 12);

  return (
    <div style={box}>
      {looksInterpolated && (
        <div style={{ ...warnBox, marginTop: 0 }}>
          ⚠ {classIds.length} ids distintos na amostra — as máscaras parecem
          interpoladas (anti-aliasing). Use máscaras com um id de classe por
          pixel (resample NEAREST).
        </div>
      )}
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8 }}>
        <span style={{ ...kicker, fontSize: 9 }}>ids nas máscaras (amostra):</span>
        {shownIds.map((id) => (
          <span
            key={id}
            style={{
              padding: "3px 9px",
              background: id >= 200 ? "oklch(0.78 0.16 75 / 0.12)" : "rgba(255,255,255,0.04)",
              border: `1px solid ${id >= 200 ? "oklch(0.78 0.16 75 / 0.5)" : "var(--vf-panel-stroke)"}`,
              borderRadius: 999,
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: id >= 200 ? warnColor : "var(--vf-text-dim)",
            }}
            title={id >= 200 ? "provável ignore_index (void)" : undefined}
          >
            {id}
          </span>
        ))}
        {stats.mask_class_ids.length > shownIds.length && (
          <span style={{ ...smallLine, fontSize: 10 }}>
            +{stats.mask_class_ids.length - shownIds.length}
          </span>
        )}
        {onApplyClasses && suggested > 0 && (
          <button
            type="button"
            onClick={() => onApplyClasses(suggested)}
            style={{
              marginLeft: "auto",
              padding: "4px 10px",
              background: "oklch(0.72 0.16 150 / 0.12)",
              border: "1px solid oklch(0.72 0.16 150 / 0.4)",
              borderRadius: 999,
              fontFamily: "var(--font-mono)",
              fontSize: 9,
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: "oklch(0.85 0.16 150)",
              cursor: "pointer",
            }}
          >
            🎯 aplicar {suggested} classes
          </button>
        )}
      </div>

      <div style={kicker}>// pareamento imagem ↔ máscara</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
        {(["train", "val", "test"] as const).map((key) => {
          const split = stats.splits[key];
          if (!split) return null;
          if (split.missing) return <MissingSplitCard key={key} label={SPLIT_LABELS[key]} />;
          const mismatch = split.unpaired_images + split.unpaired_masks > 0;
          return (
            <div key={key} style={splitCard}>
              <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
                <span style={splitLabel}>{SPLIT_LABELS[key]}</span>
                <span style={bigNumber}>{split.paired} pares</span>
              </div>
              <div style={smallLine}>
                {split.images} img · {split.masks} másc
              </div>
              {mismatch && (
                <div style={{ ...smallLine, color: warnColor }}>
                  ⚠ {split.unpaired_images} img sem máscara ·{" "}
                  {split.unpaired_masks} másc sem img
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// -------------------------------------------------------------------- anomaly

interface AnomalyDatasetStatsProps {
  baseDir: string;
  trainDir: string;
  testDir: string;
  normalDir: string;
}

export function AnomalyDatasetStats(props: AnomalyDatasetStatsProps) {
  const body = {
    base_dir: props.baseDir,
    train_dir: props.trainDir,
    test_dir: props.testDir,
    normal_dir: props.normalDir,
  };
  const stats = useDebouncedStats<AnomalyDatasetStatsResponse>(
    JSON.stringify(body),
    props.baseDir.trim() !== "",
    () => fetchAnomalyDatasetStats(body),
  );
  if (!stats) return null;
  if (stats.message) return <div style={warnBox}>{stats.message}</div>;

  const defects = Object.entries(stats.test_anomalous);
  const totalAnomalous = defects.reduce((acc, [, n]) => acc + n, 0);

  return (
    <div style={box}>
      <div style={kicker}>// distribuição normal vs. anômalo</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
        <div style={splitCard}>
          <span style={splitLabel}>treino (normal)</span>
          <span style={bigNumber}>{stats.train_normal} img</span>
        </div>
        <div style={splitCard}>
          <span style={splitLabel}>teste · normal</span>
          <span style={bigNumber}>{stats.test_normal} img</span>
        </div>
        <div style={splitCard}>
          <span style={splitLabel}>teste · anômalo</span>
          <span style={bigNumber}>{totalAnomalous} img</span>
          {stats.missing_test && (
            <span style={{ ...smallLine, color: warnColor }}>pasta de teste ausente</span>
          )}
        </div>
      </div>
      {defects.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {defects.map(([name, n]) => (
            <span
              key={name}
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
              {name}
              <span style={{ color: "var(--accent-vf)", marginLeft: 6 }}>{n}</span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ----------------------------------------------------------------- regression

interface RegressionDatasetStatsProps {
  baseDir: string;
  imagesDir: string;
  trainCsv: string;
  valCsv: string;
  testCsv: string;
  imageColumn: string;
  /** Comma-separated, as edited in the form. */
  targetColumns: string;
}

export function RegressionDatasetStats(props: RegressionDatasetStatsProps) {
  const targets = props.targetColumns
    .split(",")
    .map((c) => c.trim())
    .filter(Boolean);
  const body = {
    base_dir: props.baseDir,
    images_dir: props.imagesDir,
    train_csv: props.trainCsv,
    val_csv: props.valCsv,
    test_csv: props.testCsv,
    image_column: props.imageColumn,
    target_columns: targets.length > 0 ? targets : ["target"],
  };
  const stats = useDebouncedStats<RegressionDatasetStatsResponse>(
    JSON.stringify(body),
    props.baseDir.trim() !== "",
    () => fetchRegressionDatasetStats(body),
  );
  if (!stats) return null;
  if (stats.message) return <div style={warnBox}>{stats.message}</div>;

  const fmt = (v: number | null) =>
    v === null ? "—" : Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3);

  return (
    <div style={box}>
      <div style={kicker}>// manifest &amp; distribuição dos alvos</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 10 }}>
        {(["train", "val", "test"] as const).map((key) => {
          const split = stats.splits[key];
          if (!split) return null;
          if (split.missing) return <MissingSplitCard key={key} label={SPLIT_LABELS[key]} />;
          return (
            <div key={key} style={splitCard}>
              <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
                <span style={splitLabel}>{SPLIT_LABELS[key]}</span>
                <span style={bigNumber}>{split.rows} linhas</span>
              </div>
              {split.missing_columns.length > 0 && (
                <div style={{ ...smallLine, color: warnColor }}>
                  ⚠ colunas ausentes: {split.missing_columns.join(", ")}
                </div>
              )}
              {split.missing_images > 0 && (
                <div style={{ ...smallLine, color: warnColor }}>
                  ⚠ {split.missing_images}/{split.checked_images} imagens não
                  encontradas
                </div>
              )}
              {Object.entries(split.targets).map(([col, t]) => (
                <div key={col} style={smallLine}>
                  {col}: μ {fmt(t.mean)} · [{fmt(t.min)}, {fmt(t.max)}] · n=
                  {t.count}
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}
