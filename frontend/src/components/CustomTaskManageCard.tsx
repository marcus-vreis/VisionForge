import { useState } from "react";
import { deleteCustomTask, hideCustomTask } from "../api/client";

interface CustomTaskManageCardProps {
  taskKey: string;
  label: string;
  /** Called after the tab should stop existing, so the app can refetch tabs. */
  onRemoved: () => void;
}

const sectionLabel: React.CSSProperties = {
  fontFamily: "var(--font-mono)",
  fontSize: 10,
  letterSpacing: "0.22em",
  textTransform: "uppercase",
  color: "var(--vf-text-muted)",
};

/** Remove a custom task's tab — reversibly, or for good.
 *
 * Two actions on purpose, because "remove this task" means two different
 * things and only one is recoverable. Hiding answers "my tab bar is full" and
 * leaves the file alone. Deleting removes the Python file the researcher
 * wrote, so it asks them to type the key: a second click is not evidence of
 * intent, typing the name is.
 */
export function CustomTaskManageCard({
  taskKey,
  label,
  onRemoved,
}: CustomTaskManageCardProps) {
  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const hide = async () => {
    setBusy(true);
    setError(null);
    try {
      await hideCustomTask(taskKey);
      onRemoved();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Falha ao ocultar.");
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    setBusy(true);
    setError(null);
    try {
      await deleteCustomTask(taskKey, typed);
      onRemoved();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Falha ao excluir.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      style={{
        marginTop: 18,
        padding: 18,
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
          gap: 12,
        }}
      >
        <div style={sectionLabel}>// gerenciar esta task</div>
        <button
          type="button"
          onClick={() => {
            setOpen((o) => !o);
            setConfirming(false);
            setTyped("");
            setError(null);
          }}
          style={{
            padding: "6px 12px",
            background: "transparent",
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
          {open ? "fechar" : "⚙ opções"}
        </button>
      </div>

      {open && (
        <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 12 }}>
          <div
            style={{
              display: "flex",
              gap: 10,
              flexWrap: "wrap",
              alignItems: "center",
            }}
          >
            <button
              type="button"
              onClick={() => void hide()}
              disabled={busy}
              title="Some com a aba; o arquivo continua em user_tasks/"
              style={{
                padding: "8px 14px",
                background: "rgba(255,255,255,0.04)",
                border: "1px solid var(--vf-panel-stroke)",
                borderRadius: 8,
                color: "var(--vf-text)",
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                letterSpacing: "0.10em",
                textTransform: "uppercase",
                cursor: busy ? "wait" : "pointer",
              }}
            >
              👁 Ocultar aba
            </button>
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--vf-text-muted)",
              }}
            >
              reversível — o arquivo fica
            </span>
          </div>

          <div style={{ height: 1, background: "var(--vf-panel-stroke)" }} />

          {!confirming ? (
            <div
              style={{
                display: "flex",
                gap: 10,
                flexWrap: "wrap",
                alignItems: "center",
              }}
            >
              <button
                type="button"
                onClick={() => setConfirming(true)}
                disabled={busy}
                style={{
                  padding: "8px 14px",
                  background: "oklch(0.704 0.191 22.216 / 0.12)",
                  border: "1px solid oklch(0.78 0.16 22 / 0.6)",
                  borderRadius: 8,
                  color: "oklch(0.88 0.14 22)",
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  letterSpacing: "0.10em",
                  textTransform: "uppercase",
                  cursor: "pointer",
                }}
              >
                🗑 Excluir do disco
              </button>
              <span
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  color: "var(--vf-text-muted)",
                }}
              >
                apaga o .py que você escreveu — sem desfazer
              </span>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 11.5,
                  color: "var(--vf-text-dim)",
                  lineHeight: 1.6,
                }}
              >
                Isto remove o arquivo de <b>{label}</b> do disco. Para confirmar,
                digite a chave da task: <b style={{ color: "var(--vf-text)" }}>{taskKey}</b>
              </div>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                <input
                  type="text"
                  value={typed}
                  onChange={(e) => setTyped(e.target.value)}
                  placeholder={taskKey}
                  autoComplete="off"
                  style={{
                    flex: 1,
                    minWidth: 200,
                    padding: "9px 12px",
                    background: "rgba(0,0,0,0.35)",
                    border: "1px solid var(--vf-panel-stroke)",
                    borderRadius: 8,
                    color: "var(--vf-text)",
                    fontFamily: "var(--font-mono)",
                    fontSize: 12,
                  }}
                />
                <button
                  type="button"
                  onClick={() => void remove()}
                  disabled={busy || typed !== taskKey}
                  title={
                    typed !== taskKey ? "Digite a chave exata para habilitar" : undefined
                  }
                  style={{
                    padding: "9px 16px",
                    background:
                      typed === taskKey
                        ? "oklch(0.704 0.191 22.216 / 0.24)"
                        : "rgba(255,255,255,0.03)",
                    border: `1px solid ${typed === taskKey ? "oklch(0.78 0.16 22)" : "var(--vf-panel-stroke)"}`,
                    borderRadius: 8,
                    color:
                      typed === taskKey ? "oklch(0.95 0.10 22)" : "var(--vf-text-muted)",
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    letterSpacing: "0.10em",
                    textTransform: "uppercase",
                    fontWeight: 600,
                    cursor: typed === taskKey && !busy ? "pointer" : "not-allowed",
                  }}
                >
                  {busy ? "Excluindo…" : "Excluir definitivamente"}
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setConfirming(false);
                    setTyped("");
                  }}
                  disabled={busy}
                  style={{
                    padding: "9px 16px",
                    background: "transparent",
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
                  Cancelar
                </button>
              </div>
            </div>
          )}

          {error && (
            <div
              style={{
                padding: "8px 12px",
                background: "oklch(0.704 0.191 22.216 / 0.10)",
                border: "1px solid oklch(0.704 0.191 22.216 / 0.4)",
                borderRadius: 8,
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "oklch(0.85 0.14 22)",
              }}
            >
              {error}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
