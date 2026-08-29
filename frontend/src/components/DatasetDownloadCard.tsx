import { useState } from "react";
import {
  datasetDownload,
  pickDatasetFolder,
  type DatasetDownloadResponse,
} from "../api/client";
import {
  buildDatasetDownloadPayload,
  makeDefaultDatasetForm,
  TORCHVISION_DATASETS,
  type DatasetDownloadForm,
  type DatasetProvider,
} from "../lib/dataset-download";
import { SelectField, Segmented, TextField } from "./controls";
import { CredentialField } from "./CredentialField";

const PROVIDERS: { value: DatasetProvider; label: string }[] = [
  { value: "torchvision", label: "torchvision" },
  { value: "roboflow", label: "Roboflow" },
  { value: "kaggle", label: "Kaggle" },
  { value: "huggingface", label: "Hugging Face" },
];

const sectionLabel: React.CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  letterSpacing: "0.22em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
};

const grid: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
  gap: 14,
  marginTop: 14,
};

/** One-shot dataset download (ADR-055): pick a provider + dataset, fetch it into a
 *  local folder, then point a task's dataset field at the result. */
export function DatasetDownloadCard({
  accent = "var(--accent-vf)",
  collapsible = true,
}: {
  accent?: string;
  /** False inside the Datasets surface, where the form *is* the content and
   * a collapse toggle would just hide the only thing on screen. */
  collapsible?: boolean;
}) {
  const [open, setOpen] = useState(!collapsible);
  const [form, setForm] = useState<DatasetDownloadForm>(makeDefaultDatasetForm());
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<DatasetDownloadResponse | null>(null);
  const [msg, setMsg] = useState<{
    kind: "info" | "error" | "success";
    text: string;
  } | null>(null);

  const set = (patch: Partial<DatasetDownloadForm>) =>
    setForm((f) => ({ ...f, ...patch }));

  const pick = async () => {
    const res = await pickDatasetFolder();
    if (!res.cancelled && res.path) set({ out_dir: res.path });
  };

  const run = async () => {
    if (!form.dataset.trim() || !form.out_dir.trim()) {
      setMsg({ kind: "error", text: "Informe o dataset e a pasta de saída." });
      return;
    }
    setRunning(true);
    setResult(null);
    setMsg({ kind: "info", text: "Baixando… (pode demorar)" });
    try {
      const res = await datasetDownload(buildDatasetDownloadPayload(form));
      setResult(res);
      setMsg({
        kind: "success",
        text: `${res.total_images} imagens em ${res.out_dir}`,
      });
    } catch (e) {
      setMsg({
        kind: "error",
        text: e instanceof Error ? e.message : "Falha no download.",
      });
    } finally {
      setRunning(false);
    }
  };

  const card: React.CSSProperties = {
    background: "var(--vf-panel)",
    border: "1px solid var(--vf-panel-stroke)",
    borderRadius: 18,
    padding: 22,
    backdropFilter: "blur(14px)",
    marginTop: 18,
  };

  return (
    <div style={card}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={sectionLabel}>Baixar dataset (online → pasta local)</div>
        {collapsible && (
          <button
            type="button"
            onClick={() => {
              setOpen((o) => !o);
              setMsg(null);
            }}
            style={{
              padding: "6px 12px",
              background: "var(--accent-soft)",
              border: `1px solid ${accent}`,
              borderRadius: 8,
              color: "var(--vf-text)",
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              cursor: "pointer",
              letterSpacing: "0.10em",
              textTransform: "uppercase",
            }}
          >
            {open ? "cancelar" : "+ baixar dataset"}
          </button>
        )}
      </div>

      {open && (
        <>
          <div style={{ maxWidth: 360, marginTop: 14 }}>
            <Segmented
              label="Provedor"
              value={form.provider}
              // Trocar de provedor limpa o dataset: o campo guardava o
              // `cifar10` do torchvision e o mostrava como se fosse um
              // workspace/projeto do Roboflow, que é um jeito silencioso de
              // mandar a pessoa baixar a coisa errada.
              onChange={(v) =>
                set({
                  provider: v as DatasetProvider,
                  dataset: v === "torchvision" ? "cifar10" : "",
                })
              }
              options={PROVIDERS}
            />
          </div>

          <div style={grid}>
            {form.provider === "torchvision" ? (
              <>
                <SelectField
                  label="Dataset"
                  value={form.dataset}
                  onChange={(v) => set({ dataset: v })}
                  options={TORCHVISION_DATASETS}
                  hint="built-in"
                />
                <TextField
                  label="Limite por classe"
                  value={form.limit}
                  onChange={(v) => set({ limit: v })}
                  placeholder="vazio = tudo"
                  hint="opcional"
                  mono
                />
              </>
            ) : form.provider === "roboflow" ? (
              <>
                {/* Sem `hint`: o rótulo já é longo, e numa coluna de 200px o
                    texto da dica quebrava por cima do campo vizinho. */}
                <TextField
                  label="Dataset ou URL"
                  value={form.dataset}
                  onChange={(v) => set({ dataset: v })}
                  placeholder="workspace/projeto — ou cole a URL"
                  mono
                />
                <TextField
                  label="Versão"
                  value={form.version}
                  onChange={(v) => set({ version: v })}
                  placeholder="1"
                  mono
                />
                <CredentialField
                  provider="roboflow"
                  label="API key"
                  hint="app.roboflow.com → Settings → API Keys"
                />
                <TextField
                  label="Formato"
                  value={form.dataset_format}
                  onChange={(v) => set({ dataset_format: v })}
                  placeholder="folder"
                  hint="folder p/ classificação"
                  mono
                />
              </>
            ) : form.provider === "kaggle" ? (
              <>
                <TextField
                  label="Dataset (owner/slug)"
                  value={form.dataset}
                  onChange={(v) => set({ dataset: v })}
                  placeholder="zalando-research/fashionmnist"
                  mono
                />
                <CredentialField
                  provider="kaggle"
                  label="API token"
                  placeholder="KGAT_…"
                  hint="kaggle.com → Settings → API → Create New Token"
                />
              </>
            ) : (
              <>
                <TextField
                  label="Dataset (id do HF Hub)"
                  value={form.dataset}
                  onChange={(v) => set({ dataset: v })}
                  placeholder="owner/dataset"
                  mono
                />
                <CredentialField
                  provider="huggingface"
                  label="Token"
                  hint="opcional — só para datasets privados"
                />
              </>
            )}

            <div style={{ gridColumn: "1 / -1", display: "flex", gap: 10, alignItems: "flex-end" }}>
              <div style={{ flex: 1 }}>
                <TextField
                  label="Pasta de saída"
                  value={form.out_dir}
                  onChange={(v) => set({ out_dir: v })}
                  placeholder="…/datasets/baixado"
                  hint="onde gravar"
                  mono
                />
              </div>
              <button
                type="button"
                onClick={() => void pick()}
                style={{
                  padding: "12px 16px",
                  background: "transparent",
                  border: "1px solid var(--vf-panel-stroke)",
                  borderRadius: 10,
                  color: "var(--vf-text-dim)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 12,
                  cursor: "pointer",
                  whiteSpace: "nowrap",
                }}
              >
                📁 Escolher
              </button>
            </div>
          </div>

          <button
            type="button"
            onClick={() => void run()}
            disabled={running}
            style={{
              marginTop: 16,
              padding: "12px 20px",
              background: "var(--accent-soft)",
              border: `1px solid ${accent}`,
              borderRadius: 10,
              color: "var(--vf-text)",
              fontFamily: "var(--font-mono)",
              fontSize: 12,
              letterSpacing: "0.10em",
              textTransform: "uppercase",
              cursor: running ? "wait" : "pointer",
              opacity: running ? 0.6 : 1,
            }}
          >
            {running ? "Baixando…" : "▶ Baixar"}
          </button>

          {msg && (
            <div
              style={{
                marginTop: 12,
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                color:
                  msg.kind === "error"
                    ? "oklch(0.78 0.16 22)"
                    : msg.kind === "success"
                      ? "oklch(0.82 0.17 150)"
                      : "var(--vf-text-muted)",
                whiteSpace: "pre-wrap",
              }}
            >
              {msg.text}
            </div>
          )}

          {result && Object.keys(result.splits).length > 0 && (
            <div
              style={{
                marginTop: 8,
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--vf-text-dim)",
              }}
            >
              {Object.entries(result.splits)
                .map(([s, n]) => `${s}: ${n}`)
                .join(" · ")}
              {result.classes.length > 0 && ` · ${result.classes.length} classes`}
            </div>
          )}
        </>
      )}
    </div>
  );
}
