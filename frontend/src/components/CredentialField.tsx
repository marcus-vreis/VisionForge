import { useEffect, useState } from "react";
import {
  fetchCredentials,
  forgetCredential,
  saveCredential,
  type CredentialEntry,
} from "../api/client";
import { TextField } from "./controls";

interface CredentialFieldProps {
  provider: "roboflow" | "kaggle" | "huggingface";
  label: string;
  hint?: string;
  placeholder?: string;
  /** Bumped by the parent to force a re-read after another field saved. */
  refreshToken?: number;
}

type Feedback = { kind: "ok" | "error"; text: string } | null;

/** A provider key you type once.
 *
 * When a key is already stored the field starts empty and the label carries
 * the masked value, so the state is visible without the secret being on
 * screen — a screenshot of this panel is not a leak. Typing a new value and
 * pressing Salvar replaces it; the download uses the stored key whenever the
 * field is left blank.
 */
export function CredentialField({
  provider,
  label,
  hint,
  placeholder,
  refreshToken = 0,
}: CredentialFieldProps) {
  const [value, setValue] = useState("");
  const [entry, setEntry] = useState<CredentialEntry | null>(null);
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState<Feedback>(null);

  useEffect(() => {
    fetchCredentials()
      .then((r) => setEntry(r.providers[provider] ?? null))
      .catch(() => setEntry(null));
  }, [provider, refreshToken]);

  // A stale "saved!" under a field the user has since edited is a lie.
  useEffect(() => {
    if (value) setFeedback(null);
  }, [value]);

  const save = async () => {
    if (!value.trim()) {
      setFeedback({ kind: "error", text: "Digite a chave antes de salvar." });
      return;
    }
    setBusy(true);
    try {
      const r = await saveCredential(provider, value.trim());
      setEntry(r.providers[provider] ?? null);
      setValue("");
      setFeedback({ kind: "ok", text: "Salva — não precisa digitar de novo." });
    } catch (e) {
      setFeedback({
        kind: "error",
        text: e instanceof Error ? e.message : "Falha ao salvar.",
      });
    } finally {
      setBusy(false);
    }
  };

  const forget = async () => {
    setBusy(true);
    try {
      const r = await forgetCredential(provider);
      setEntry(r.providers[provider] ?? null);
      setFeedback({ kind: "ok", text: "Chave removida deste computador." });
    } catch (e) {
      setFeedback({
        kind: "error",
        text: e instanceof Error ? e.message : "Falha ao remover.",
      });
    } finally {
      setBusy(false);
    }
  };

  const saved = entry?.saved ?? false;

  return (
    <div style={{ gridColumn: "1 / -1" }}>
      <TextField
        label={label}
        value={value}
        onChange={setValue}
        placeholder={saved ? `salva: ${entry?.masked}` : (placeholder ?? "")}
        hint={saved ? "deixe em branco para usar a salva" : hint}
        mono
      />
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginTop: 8,
          flexWrap: "wrap",
        }}
      >
        <button
          type="button"
          onClick={() => void save()}
          disabled={busy}
          style={{
            padding: "6px 12px",
            background: "var(--accent-soft)",
            border: "1px solid var(--accent-vf)",
            borderRadius: 8,
            color: "var(--vf-text)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            letterSpacing: "0.10em",
            textTransform: "uppercase",
            cursor: busy ? "wait" : "pointer",
            opacity: busy ? 0.6 : 1,
          }}
        >
          {saved ? "↻ Substituir" : "💾 Salvar"}
        </button>
        {saved && (
          <button
            type="button"
            onClick={() => void forget()}
            disabled={busy}
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
              cursor: busy ? "wait" : "pointer",
            }}
          >
            Esquecer
          </button>
        )}
        {feedback && (
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              color:
                feedback.kind === "ok"
                  ? "oklch(0.88 0.16 150)"
                  : "oklch(0.85 0.14 22)",
            }}
          >
            {feedback.kind === "ok" ? "✓ " : "⚠ "}
            {feedback.text}
          </span>
        )}
      </div>
    </div>
  );
}
