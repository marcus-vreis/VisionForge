import { useEffect, useState } from "react";
import {
  ApiError,
  artifactUrl,
  fetchRunDetail,
  pickDatasetFolder,
  testRunOnDataset,
  type RunDetail,
  type TestRecord,
} from "../api/client";
import { Lightbox } from "./Lightbox";

interface RunDetailPanelProps {
  runId: string;
  onBack: () => void;
}

/** Plot file naming convention from the backend; used to humanize labels. */
const GRAPH_LABELS: Record<string, string> = {
  "loss.png": "Loss (train + val)",
  "accuracy.png": "Accuracy (train + val)",
  "confusion_matrix.png": "Matriz de confusão",
  "confusion_matrix_normalized.png": "Matriz de confusão (normalizada)",
  "roc_curve.png": "Curva ROC",
  "precision_recall_curve.png": "Curva Precision-Recall",
};

function metricLabel(key: string): string {
  const labels: Record<string, string> = {
    accuracy: "Acurácia",
    f1: "F1",
    precision: "Precisão",
    recall: "Recall",
    auc_roc: "AUC-ROC",
    test_accuracy: "Acurácia (teste)",
    test_f1: "F1 (teste)",
    test_precision: "Precisão (teste)",
    test_recall: "Recall (teste)",
    test_auc_roc: "AUC-ROC (teste)",
    best_val_loss: "Melhor val loss",
    best_epoch: "Melhor epoch",
    total_epochs: "Epochs treinados",
  };
  return labels[key] ?? key;
}

function fmtMetric(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return v % 1 === 0 ? String(v) : v.toFixed(4);
  return String(v);
}

export function RunDetailPanel({ runId, onBack }: RunDetailPanelProps) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lightbox, setLightbox] = useState<{ src: string; caption: string } | null>(null);
  const [testForm, setTestForm] = useState({
    base_dir: "",
    train_dir: "train",
    val_dir: "val",
    test_dir: "test",
    label: "",
  });
  const [testing, setTesting] = useState(false);
  const [testMsg, setTestMsg] = useState<{ kind: "info" | "error" | "success"; text: string } | null>(null);
  const [showTestForm, setShowTestForm] = useState(false);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    fetchRunDetail(runId)
      .then((d) => {
        if (alive) setDetail(d);
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setError(e instanceof Error ? e.message : "Falha ao carregar detalhes.");
      })
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [runId]);

  const reload = async () => {
    try {
      const d = await fetchRunDetail(runId);
      setDetail(d);
    } catch {
      /* ignore — keep existing detail */
    }
  };

  const pickFolder = async () => {
    setTestMsg({ kind: "info", text: "Abrindo seletor…" });
    try {
      const res = await pickDatasetFolder();
      if (res.cancelled) {
        setTestMsg({ kind: "info", text: res.message ?? "Cancelado." });
        return;
      }
      setTestForm((f) => ({ ...f, base_dir: res.path }));
      setTestMsg({ kind: "success", text: `Pasta: ${res.path}` });
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Falha ao escolher pasta.";
      setTestMsg({ kind: "error", text: msg });
    }
  };

  const runTest = async () => {
    if (!testForm.base_dir.trim()) {
      setTestMsg({ kind: "error", text: "Informe o diretório base do dataset de teste." });
      return;
    }
    setTesting(true);
    setTestMsg({ kind: "info", text: "Avaliando modelo no novo dataset…" });
    try {
      const record = await testRunOnDataset(runId, {
        base_dir: testForm.base_dir,
        train_dir: testForm.train_dir || "train",
        val_dir: testForm.val_dir || "val",
        test_dir: testForm.test_dir || "test",
        label: testForm.label || undefined,
      });
      setTestMsg({
        kind: "success",
        text: `Teste registrado: ${record.test_id}`,
      });
      setShowTestForm(false);
      await reload();
    } catch (e) {
      const msg =
        e instanceof ApiError ? e.message : e instanceof Error ? e.message : "Falha no teste.";
      setTestMsg({ kind: "error", text: msg });
    } finally {
      setTesting(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <button
          type="button"
          onClick={onBack}
          style={{
            padding: "6px 12px",
            background: "rgba(255,255,255,0.04)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 8,
            color: "var(--vf-text-dim)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            letterSpacing: "0.10em",
            textTransform: "uppercase",
            cursor: "pointer",
          }}
        >
          ← histórico
        </button>
        <div style={{ fontFamily: "var(--font-mono)", fontSize: 14, color: "var(--vf-text)" }}>
          {runId}
        </div>
      </div>

      {loading && (
        <div style={{ padding: 32, textAlign: "center", color: "var(--vf-text-muted)" }}>
          carregando…
        </div>
      )}

      {error && (
        <div
          style={{
            padding: 14,
            background: "oklch(0.704 0.191 22.216 / 0.10)",
            border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
            borderRadius: 10,
            color: "oklch(0.85 0.14 22)",
            fontFamily: "var(--font-mono)",
            fontSize: 12,
          }}
        >
          {error}
        </div>
      )}

      {detail && (
        <>
          <Section title="Localização no disco">
            <PathRow label="Pasta do run" value={detail.run_dir} />
            {detail.artifacts.model && (
              <PathRow label="Checkpoint" value={detail.artifacts.model} />
            )}
            {detail.device_used && (
              <KeyRow label="Dispositivo usado" value={detail.device_used} />
            )}
          </Section>

          <Section title="Métricas">
            <MetricsGrid metrics={detail.metrics} />
          </Section>

          {detail.artifacts.graphics && detail.artifacts.graphics.length > 0 && (
            <Section title="Gráficos (clique para expandir)">
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
                  gap: 12,
                }}
              >
                {detail.artifacts.graphics.map((g) => {
                  const filename = g.replace(/\\/g, "/").split("/").pop() ?? g;
                  const label = GRAPH_LABELS[filename] ?? filename;
                  const url = artifactUrl(g);
                  return (
                    <button
                      key={g}
                      type="button"
                      onClick={() => setLightbox({ src: url, caption: g })}
                      style={{
                        background: "rgba(0,0,0,0.3)",
                        border: "1px solid var(--vf-panel-stroke)",
                        borderRadius: 10,
                        padding: 0,
                        overflow: "hidden",
                        cursor: "zoom-in",
                        display: "flex",
                        flexDirection: "column",
                        gap: 6,
                        textAlign: "left",
                      }}
                    >
                      <img
                        src={url}
                        alt={label}
                        style={{ width: "100%", height: "auto", display: "block" }}
                      />
                      <div
                        style={{
                          padding: "6px 10px 8px",
                          fontFamily: "var(--font-mono)",
                          fontSize: 11,
                          color: "var(--vf-text-dim)",
                        }}
                      >
                        {label}
                        <div
                          style={{
                            fontSize: 9,
                            color: "var(--vf-text-muted)",
                            wordBreak: "break-all",
                            marginTop: 2,
                          }}
                        >
                          {g}
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </Section>
          )}

          <Section
            title="Testes neste modelo"
            action={
              <button
                type="button"
                onClick={() => setShowTestForm((s) => !s)}
                style={{
                  padding: "6px 12px",
                  background: "var(--accent-soft)",
                  border: "1px solid var(--accent-vf)",
                  borderRadius: 8,
                  color: "var(--vf-text)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  cursor: "pointer",
                  letterSpacing: "0.10em",
                  textTransform: "uppercase",
                }}
              >
                {showTestForm ? "cancelar" : "+ testar"}
              </button>
            }
          >
            {showTestForm && (
              <div
                style={{
                  padding: 14,
                  border: "1px dashed var(--vf-panel-stroke)",
                  borderRadius: 10,
                  marginBottom: 12,
                  display: "flex",
                  flexDirection: "column",
                  gap: 10,
                }}
              >
                <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
                  <FormField
                    label="Diretório base"
                    value={testForm.base_dir}
                    onChange={(v) => setTestForm((f) => ({ ...f, base_dir: v }))}
                    placeholder="ex: C:/datasets/coffee_v2"
                  />
                  <button
                    type="button"
                    onClick={() => void pickFolder()}
                    style={{
                      padding: "10px 14px",
                      background: "transparent",
                      border: "1px solid var(--vf-panel-stroke)",
                      borderRadius: 10,
                      color: "var(--vf-text-dim)",
                      fontFamily: "var(--font-mono)",
                      fontSize: 11,
                      cursor: "pointer",
                      whiteSpace: "nowrap",
                    }}
                  >
                    📁 Escolher
                  </button>
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(3, 1fr)",
                    gap: 10,
                  }}
                >
                  <FormField
                    label="Subpasta treino"
                    value={testForm.train_dir}
                    onChange={(v) => setTestForm((f) => ({ ...f, train_dir: v }))}
                  />
                  <FormField
                    label="Subpasta validação"
                    value={testForm.val_dir}
                    onChange={(v) => setTestForm((f) => ({ ...f, val_dir: v }))}
                  />
                  <FormField
                    label="Subpasta teste"
                    value={testForm.test_dir}
                    onChange={(v) => setTestForm((f) => ({ ...f, test_dir: v }))}
                  />
                </div>
                <FormField
                  label="Rótulo (opcional)"
                  value={testForm.label}
                  onChange={(v) => setTestForm((f) => ({ ...f, label: v }))}
                  placeholder="ex: holdout_2026"
                />
                <button
                  type="button"
                  onClick={() => void runTest()}
                  disabled={testing}
                  style={{
                    padding: "12px 20px",
                    background:
                      "linear-gradient(180deg, var(--accent-soft) 0%, rgba(8,10,14,0.4) 100%)",
                    border: "1px solid var(--accent-vf)",
                    borderRadius: 10,
                    color: "var(--vf-text)",
                    fontFamily: "var(--font-mono)",
                    fontSize: 12,
                    fontWeight: 600,
                    letterSpacing: "0.10em",
                    textTransform: "uppercase",
                    cursor: testing ? "wait" : "pointer",
                    opacity: testing ? 0.6 : 1,
                  }}
                >
                  {testing ? "Avaliando…" : "▶ Rodar teste"}
                </button>
                {testMsg && (
                  <div
                    style={{
                      padding: "8px 12px",
                      fontFamily: "var(--font-mono)",
                      fontSize: 11,
                      borderRadius: 8,
                      background:
                        testMsg.kind === "error"
                          ? "oklch(0.704 0.191 22.216 / 0.10)"
                          : testMsg.kind === "success"
                            ? "oklch(0.72 0.16 150 / 0.10)"
                            : "rgba(255,255,255,0.04)",
                      color:
                        testMsg.kind === "error"
                          ? "oklch(0.85 0.14 22)"
                          : testMsg.kind === "success"
                            ? "oklch(0.85 0.16 150)"
                            : "var(--vf-text-dim)",
                    }}
                  >
                    {testMsg.text}
                  </div>
                )}
              </div>
            )}

            {detail.tests.length === 0 ? (
              <div
                style={{
                  padding: 18,
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  color: "var(--vf-text-muted)",
                  textAlign: "center",
                  border: "1px dashed var(--vf-panel-stroke)",
                  borderRadius: 10,
                }}
              >
                Nenhum teste executado ainda neste modelo.
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {detail.tests.map((t) => (
                  <TestRow key={t.test_id} test={t} onOpenImage={(src, caption) => setLightbox({ src, caption })} />
                ))}
              </div>
            )}
          </Section>
        </>
      )}

      {lightbox && (
        <Lightbox
          src={lightbox.src}
          caption={lightbox.caption}
          onClose={() => setLightbox(null)}
        />
      )}
    </div>
  );
}

interface SectionProps {
  title: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}

function Section({ title, children, action }: SectionProps) {
  return (
    <div
      style={{
        padding: "14px 16px",
        background: "rgba(255,255,255,0.02)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 12,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 10,
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
          {title}
        </div>
        {action}
      </div>
      {children}
    </div>
  );
}

function PathRow({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "6px 0",
        borderTop: "1px solid var(--vf-panel-stroke)",
        marginTop: 6,
      }}
    >
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--vf-text-muted)",
          letterSpacing: "0.10em",
          textTransform: "uppercase",
          minWidth: 110,
        }}
      >
        {label}
      </span>
      <code
        style={{
          flex: 1,
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          color: "var(--vf-text)",
          wordBreak: "break-all",
        }}
      >
        {value}
      </code>
      <button
        type="button"
        onClick={() => void navigator.clipboard.writeText(value)}
        title="Copiar caminho"
        style={{
          padding: "4px 8px",
          background: "rgba(255,255,255,0.04)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 6,
          color: "var(--vf-text-dim)",
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          cursor: "pointer",
        }}
      >
        copy
      </button>
    </div>
  );
}

function KeyRow({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        display: "flex",
        gap: 12,
        padding: "6px 0",
        borderTop: "1px solid var(--vf-panel-stroke)",
        marginTop: 6,
      }}
    >
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--vf-text-muted)",
          letterSpacing: "0.10em",
          textTransform: "uppercase",
          minWidth: 110,
        }}
      >
        {label}
      </span>
      <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--vf-text)" }}>
        {value}
      </span>
    </div>
  );
}

function MetricsGrid({ metrics }: { metrics: Record<string, unknown> }) {
  const entries = Object.entries(metrics);
  if (entries.length === 0) {
    return <div style={{ color: "var(--vf-text-muted)", fontSize: 12 }}>Sem métricas registradas.</div>;
  }
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
        gap: 8,
      }}
    >
      {entries.map(([k, v]) => (
        <div
          key={k}
          style={{
            padding: "8px 10px",
            background: "rgba(0,0,0,0.3)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 8,
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
            {metricLabel(k)}
          </div>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 15,
              fontWeight: 600,
              color: "var(--vf-text)",
              marginTop: 2,
            }}
          >
            {fmtMetric(v)}
          </div>
        </div>
      ))}
    </div>
  );
}

interface TestRowProps {
  test: TestRecord;
  onOpenImage: (src: string, caption: string) => void;
}

function TestRow({ test, onOpenImage }: TestRowProps) {
  return (
    <div
      style={{
        padding: "10px 14px",
        background: "rgba(255,255,255,0.025)",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 10,
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontWeight: 600, fontSize: 13 }}>{test.label}</span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--vf-text-muted)" }}>
          {new Date(test.timestamp).toLocaleString("pt-BR")}
        </span>
      </div>
      <code
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--vf-text-dim)",
          wordBreak: "break-all",
        }}
      >
        {test.base_dir}
      </code>
      <MetricsGrid metrics={test.metrics} />
      {Object.entries(test.artifacts).length > 0 && (
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {Object.entries(test.artifacts).map(([k, p]) => (
            <button
              key={k}
              type="button"
              onClick={() => onOpenImage(artifactUrl(p), p)}
              style={{
                padding: "4px 10px",
                background: "rgba(255,255,255,0.04)",
                border: "1px solid var(--vf-panel-stroke)",
                borderRadius: 6,
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                color: "var(--vf-text-dim)",
                cursor: "pointer",
              }}
            >
              📊 {k}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

interface FormFieldProps {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}

function FormField({ label, value, onChange, placeholder }: FormFieldProps) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1 }}>
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
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        style={{
          padding: "8px 10px",
          background: "rgba(0,0,0,0.35)",
          border: "1px solid var(--vf-panel-stroke)",
          borderRadius: 8,
          color: "var(--vf-text)",
          fontFamily: "var(--font-mono)",
          fontSize: 12,
        }}
      />
    </label>
  );
}
