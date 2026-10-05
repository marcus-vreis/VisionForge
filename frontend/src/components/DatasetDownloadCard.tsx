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
import { useT } from "../i18n/useT";

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
  const t = useT();
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
      setMsg({ kind: "error", text: t.datasetDownload.needDatasetAndFolder });
      return;
    }
    setRunning(true);
    setResult(null);
    setMsg({ kind: "info", text: t.datasetDownload.downloadingWait });
    try {
      const res = await datasetDownload(buildDatasetDownloadPayload(form));
      setResult(res);
      setMsg({
        kind: "success",
        text: t.datasetDownload.done(res.total_images, res.out_dir),
      });
    } catch (e) {
      setMsg({
        kind: "error",
        text: e instanceof Error ? e.message : t.datasetDownload.failed,
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
        <div style={sectionLabel}>{t.datasetDownload.title}</div>
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
            {open ? t.datasetDownload.cancel : t.datasetDownload.open}
          </button>
        )}
      </div>

      {open && (
        <>
          <div style={{ maxWidth: 360, marginTop: 14 }}>
            <Segmented
              label={t.datasetDownload.provider}
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
                  label={t.datasetDownload.torchvision.dataset}
                  value={form.dataset}
                  onChange={(v) => set({ dataset: v })}
                  options={TORCHVISION_DATASETS}
                  hint={t.datasetDownload.torchvision.datasetHint}
                />
                <TextField
                  label={t.datasetDownload.torchvision.limit}
                  value={form.limit}
                  onChange={(v) => set({ limit: v })}
                  placeholder={t.datasetDownload.torchvision.limitPlaceholder}
                  hint={t.datasetDownload.torchvision.limitHint}
                  mono
                />
              </>
            ) : form.provider === "roboflow" ? (
              <>
                {/* Sem `hint`: o rótulo já é longo, e numa coluna de 200px o
                    texto da dica quebrava por cima do campo vizinho. */}
                <TextField
                  label={t.datasetDownload.roboflow.dataset}
                  value={form.dataset}
                  onChange={(v) => set({ dataset: v })}
                  placeholder={t.datasetDownload.roboflow.datasetPlaceholder}
                  mono
                />
                <TextField
                  label={t.datasetDownload.roboflow.version}
                  value={form.version}
                  onChange={(v) => set({ version: v })}
                  placeholder="1"
                  mono
                />
                <CredentialField
                  provider="roboflow"
                  label={t.datasetDownload.roboflow.apiKey}
                  hint={t.datasetDownload.roboflow.apiKeyHint}
                />
                <TextField
                  label={t.datasetDownload.roboflow.format}
                  value={form.dataset_format}
                  onChange={(v) => set({ dataset_format: v })}
                  placeholder={t.datasetDownload.roboflow.formatPlaceholder}
                  hint={t.datasetDownload.roboflow.formatHint}
                  mono
                />
              </>
            ) : form.provider === "kaggle" ? (
              <>
                <TextField
                  label={t.datasetDownload.kaggle.dataset}
                  value={form.dataset}
                  onChange={(v) => set({ dataset: v })}
                  placeholder={t.datasetDownload.kaggle.datasetPlaceholder}
                  mono
                />
                <CredentialField
                  provider="kaggle"
                  label={t.datasetDownload.kaggle.token}
                  placeholder={t.datasetDownload.kaggle.tokenPlaceholder}
                  hint={t.datasetDownload.kaggle.tokenHint}
                />
              </>
            ) : (
              <>
                <TextField
                  label={t.datasetDownload.huggingface.dataset}
                  value={form.dataset}
                  onChange={(v) => set({ dataset: v })}
                  placeholder={t.datasetDownload.huggingface.datasetPlaceholder}
                  mono
                />
                <CredentialField
                  provider="huggingface"
                  label={t.datasetDownload.huggingface.token}
                  hint={t.datasetDownload.huggingface.tokenHint}
                />
              </>
            )}

            <div style={{ gridColumn: "1 / -1", display: "flex", gap: 10, alignItems: "flex-end" }}>
              <div style={{ flex: 1 }}>
                <TextField
                  label={t.datasetDownload.outDir}
                  value={form.out_dir}
                  onChange={(v) => set({ out_dir: v })}
                  placeholder={t.datasetDownload.outDirPlaceholder}
                  hint={t.datasetDownload.outDirHint}
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
                {t.datasetDownload.browse}
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
            {running ? t.datasetDownload.downloading : t.datasetDownload.download}
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
              {result.classes.length > 0 && t.datasetDownload.classes(result.classes.length)}
            </div>
          )}
        </>
      )}
    </div>
  );
}
