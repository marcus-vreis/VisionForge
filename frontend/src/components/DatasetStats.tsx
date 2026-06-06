import { useEffect, useRef, useState } from "react";
import {
  datasetFileUrl,
  fetchDatasetSamples,
  fetchDatasetStats,
  type DatasetSamplesResponse,
  type DatasetStatsResponse,
} from "../api/client";

interface DatasetStatsProps {
  baseDir: string;
  trainDir: string;
  valDir: string;
  testDir: string;
  /** Optional handler — applies detected class count to the experiment config.
   *  When ``numClasses === 2`` the parent should also flip ``task`` to "binary"
   *  (which the Pydantic validator lowers to ``num_classes=1`` internally). */
  onApplyClasses?: (numClasses: number, classNames: string[]) => void;
}

const SPLIT_LABELS: Record<string, string> = {
  train: "treino",
  val: "validação",
  test: "teste",
};

/** Compact pre-training dataset overview: image counts + imbalance flag. */
export function DatasetStats({ baseDir, trainDir, valDir, testDir, onApplyClasses }: DatasetStatsProps) {
  const [stats, setStats] = useState<DatasetStatsResponse | null>(null);
  const [samples, setSamples] = useState<DatasetSamplesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  // Guarda o último base_dir cujas classes já foram auto-aplicadas, para não
  // sobrescrever ajustes manuais a cada refetch (mudar subpasta também refetcha).
  const appliedForBaseDir = useRef<string | null>(null);

  useEffect(() => {
    if (!baseDir.trim()) {
      // Defer the reset so it does not run synchronously in the effect body.
      const t = setTimeout(() => {
        setStats(null);
        setSamples(null);
      }, 0);
      return () => clearTimeout(t);
    }
    let alive = true;
    // Defer the loading flag out of the synchronous effect body; cleared in
    // finally so a fast resolve never leaves it stuck on.
    const loadingTimer = setTimeout(() => {
      if (alive) setLoading(true);
    }, 0);
    Promise.all([
      fetchDatasetStats(baseDir, {
        train_dir: trainDir || "train",
        val_dir: valDir || "val",
        test_dir: testDir || "test",
      }),
      fetchDatasetSamples(baseDir, "train", 4).catch(() => null),
    ])
      .then(([s, sm]) => {
        if (!alive) return;
        setStats(s);
        setSamples(sm);
        // Auto-aplica o nº de classes detectado — uma vez por base_dir.
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
        if (alive) {
          setStats(null);
          setSamples(null);
        }
      })
      .finally(() => {
        clearTimeout(loadingTimer);
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
      clearTimeout(loadingTimer);
    };
  }, [baseDir, trainDir, valDir, testDir]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!baseDir.trim()) return null;
  if (loading && !stats) {
    return (
      <div
        style={{
          marginTop: 12,
          padding: "10px 14px",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          color: "var(--vf-text-muted)",
          border: "1px dashed var(--vf-panel-stroke)",
          borderRadius: 10,
        }}
      >
        Analisando dataset…
      </div>
    );
  }
  if (!stats) return null;

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
        {stats.message ?? "Nenhuma classe encontrada."}
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
      {/* Class index map — ImageFolder sorts alphabetically, so 0 = first */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          gap: 8,
        }}
      >
        <span
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 9,
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            color: "var(--vf-text-muted)",
          }}
        >
          mapeamento (ImageFolder):
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
            title={
              stats.class_names.length === 2
                ? "Detectado e aplicado ao config: task=binary, num_classes=1"
                : `Detectado e aplicado ao config: task=multiclass, num_classes=${stats.class_names.length}`
            }
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
            ✓ {stats.class_names.length === 2 ? "binary aplicado" : `multiclass · ${stats.class_names.length} aplicado`}
          </span>
        )}
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "var(--vf-text-muted)",
          }}
        >
          // distribuição do dataset
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

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 10,
        }}
      >
        {(["train", "val", "test"] as const).map((splitKey) => {
          const split = stats.splits[splitKey];
          if (!split) return null;
          return (
            <SplitCard
              key={splitKey}
              label={SPLIT_LABELS[splitKey]}
              split={split}
              classNames={stats.class_names}
            />
          );
        })}
      </div>

      {samples && Object.keys(samples.samples).length > 0 && (
        <div
          style={{
            marginTop: 4,
            paddingTop: 12,
            borderTop: "1px solid var(--vf-panel-stroke)",
            display: "flex",
            flexDirection: "column",
            gap: 8,
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
            // amostras (split: {samples.split}) — sanity-check de labels
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {Object.entries(samples.samples).map(([cn, paths]) => (
              <ClassSampleRow key={cn} className={cn} paths={paths} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

interface ClassSampleRowProps {
  className: string;
  paths: string[];
}

function ClassSampleRow({ className, paths }: ClassSampleRowProps) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <div
        style={{
          minWidth: 80,
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          color: "var(--vf-text-dim)",
          wordBreak: "break-all",
        }}
      >
        {className}
      </div>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        {paths.map((p) => (
          <img
            key={p}
            src={datasetFileUrl(p)}
            alt={`${className} sample`}
            title={p}
            style={{
              width: 56,
              height: 56,
              objectFit: "cover",
              borderRadius: 6,
              border: "1px solid var(--vf-panel-stroke)",
              background: "rgba(0,0,0,0.30)",
            }}
            onError={(e) => {
              (e.currentTarget as HTMLImageElement).style.opacity = "0.25";
            }}
          />
        ))}
        {paths.length === 0 && (
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "var(--vf-text-muted)",
              opacity: 0.6,
            }}
          >
            sem imagens
          </div>
        )}
      </div>
    </div>
  );
}

interface SplitCardProps {
  label: string;
  split: { total_images: number; classes: Record<string, number>; missing: boolean };
  classNames: string[];
}

function SplitCard({ label, split, classNames }: SplitCardProps) {
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

  const total = split.total_images;
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
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
        }}
      >
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
          {total}
        </span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        {classNames.map((cn) => {
          const count = split.classes[cn] ?? 0;
          const pct = total === 0 ? 0 : (count / total) * 100;
          return (
            <div
              key={cn}
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 2,
              }}
            >
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
