import { useEffect, useState } from "react";
import { fetchDatasetStats, type DatasetStatsResponse } from "../api/client";

interface DatasetStatsProps {
  baseDir: string;
  trainDir: string;
  valDir: string;
  testDir: string;
}

const SPLIT_LABELS: Record<string, string> = {
  train: "treino",
  val: "validação",
  test: "teste",
};

/** Compact pre-training dataset overview: image counts + imbalance flag. */
export function DatasetStats({ baseDir, trainDir, valDir, testDir }: DatasetStatsProps) {
  const [stats, setStats] = useState<DatasetStatsResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!baseDir.trim()) {
      setStats(null);
      return;
    }
    let alive = true;
    setLoading(true);
    fetchDatasetStats(baseDir, {
      train_dir: trainDir || "train",
      val_dir: valDir || "val",
      test_dir: testDir || "test",
    })
      .then((d) => alive && setStats(d))
      .catch(() => alive && setStats(null))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [baseDir, trainDir, valDir, testDir]);

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
