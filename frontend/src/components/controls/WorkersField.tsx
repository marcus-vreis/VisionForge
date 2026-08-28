import { useEffect, useState } from "react";

import { fetchSystemInfo } from "../../api/client";
import { paramHelp } from "../../lib/param-help";
import { InfoDot } from "./InfoDot";

/** The worker count, automatic by default and only a number when asked for.
 *
 * `-1` is the config's "decide for me" (ADR-103): the backend measures the free
 * commit charge at load time and divides it by what a worker costs on this
 * machine. That is a better number than anything the browser can compute, and
 * the field used to make the researcher go find it — the "auto" button copied a
 * suggestion into the box, so the value froze at whatever the machine had free
 * on the day the form was filled in.
 *
 * So automatic is the resting state and shows what it resolved to; the number
 * only appears when someone deliberately takes it over.
 */
export function WorkersField({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  const [suggested, setSuggested] = useState<number | null>(null);
  const auto = value < 0;

  useEffect(() => {
    let alive = true;
    fetchSystemInfo()
      .then((info) => alive && setSuggested(info.suggested_workers))
      .catch(() => {
        /* The probe is a courtesy: the backend resolves -1 either way. */
      });
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 9,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: "var(--vf-text-muted)",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <span>Workers</span>
        <InfoDot text={paramHelp("num_workers") ?? ""} />
        <button
          type="button"
          onClick={() => onChange(auto ? (suggested ?? 2) : -1)}
          style={{
            marginLeft: "auto",
            padding: "2px 8px",
            background: auto ? "var(--accent-soft)" : "rgba(255,255,255,0.04)",
            border: `1px solid ${auto ? "var(--accent-vf)" : "var(--vf-panel-stroke)"}`,
            borderRadius: 6,
            color: auto ? "var(--vf-text)" : "var(--vf-text-dim)",
            fontFamily: "var(--font-mono)",
            fontSize: 9,
            letterSpacing: "0.10em",
            textTransform: "uppercase",
            cursor: "pointer",
          }}
          title={
            auto
              ? "Definir o número manualmente"
              : "Voltar a decidir pela memória livre da máquina"
          }
        >
          auto
        </button>
      </span>
      {auto ? (
        <div
          style={{
            padding: "8px 12px",
            background: "rgba(0,0,0,0.35)",
            border: "1px dashed var(--vf-panel-stroke)",
            borderRadius: 8,
            fontFamily: "var(--font-mono)",
            fontSize: 13,
            color: "var(--vf-text-dim)",
            display: "flex",
            alignItems: "baseline",
            gap: 8,
          }}
        >
          <span>automático</span>
          {suggested !== null && (
            <span style={{ fontSize: 11, color: "var(--vf-text-muted)" }}>
              ≈ {suggested} agora
            </span>
          )}
        </div>
      ) : (
        <input
          type="number"
          value={value}
          min={0}
          step={1}
          onChange={(e) => {
            const v = parseInt(e.target.value, 10);
            if (!Number.isNaN(v)) onChange(v);
          }}
          style={{
            padding: "8px 12px",
            background: "rgba(0,0,0,0.30)",
            border: "1px solid var(--vf-panel-stroke)",
            borderRadius: 8,
            fontFamily: "var(--font-mono)",
            fontSize: 13,
            color: "var(--vf-text)",
          }}
        />
      )}
    </div>
  );
}
