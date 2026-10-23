import { useRef, useState } from "react";
import { auditDataset, type DatasetAuditReport } from "../api/client";
import { useT } from "../i18n/useT";
import { TextField, Toggle } from "./controls";

interface DatasetAuditProps {
  baseDir: string;
  trainDir: string;
  valDir: string;
  testDir: string;
}

/** Show measured audit counts and bounded, escaped file details. */
export function DatasetAuditResult({ report }: { report: DatasetAuditReport }) {
  const t = useT();
  const e = t.evidence;
  return (
    <div role="status" data-testid="audit-result">
      <p>{report.ok ? e.clean : e.findings}</p>
      <p>{e.scope}</p>
      {Object.entries(report.image_counts).map(([split, count]) => (
        <div key={split}>{e.imageCount(split, count)}</div>
      ))}
      <ul>
        <li>{e.duplicateCount(report.duplicate_groups.length)}</li>
        <li>{e.missingCount(report.missing_splits.length)}</li>
        <li>{e.emptyCount(report.empty_splits.length)}</li>
        <li>{e.unreadableCount(report.unreadable_count)}</li>
        <li>{e.groupCount(report.group_overlaps.length)}</li>
      </ul>
      {!report.ok && (
        <details>
          <summary>{e.details}</summary>
          <ul style={{ overflowWrap: "anywhere" }}>
            {report.duplicate_groups.slice(0, 5).map((group) => (
              <li key={group.sha256}>
                {group.sha256} · {group.splits.join(", ")}
                <ul>{group.paths.slice(0, 5).map((path) => <li key={path}>{path}</li>)}</ul>
              </li>
            ))}
            {report.missing_splits.slice(0, 5).map((split) => <li key={`missing-${split}`}>{e.missingCount(1)} · {split}</li>)}
            {report.empty_splits.slice(0, 5).map((split) => <li key={`empty-${split}`}>{e.emptyCount(1)} · {split}</li>)}
            {report.unreadable.slice(0, 5).map((item) => <li key={item.path}>{item.path} · {item.reason}</li>)}
            {report.group_overlaps.slice(0, 5).map((group) => (
              <li key={group.group}>
                {group.group} · {group.splits.join(", ")}
                <ul>{group.paths.slice(0, 5).map((path) => <li key={path}>{path}</li>)}</ul>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

/** Run an image audit only on request, against the current configured splits. */
export function DatasetAudit({ baseDir, trainDir, valDir, testDir }: DatasetAuditProps) {
  const t = useT();
  const [manifest, setManifest] = useState("");
  const [strict, setStrict] = useState(false);
  const [state, setState] = useState<{
    key: string;
    loading: boolean;
    report?: DatasetAuditReport;
    error?: string;
  } | null>(null);
  const sequence = useRef(0);
  const body = {
    base_dir: baseDir,
    splits: { train: trainDir, val: valDir, test: testDir },
    group_manifest: manifest.trim() || null,
    strict_groups: strict,
  };
  const key = JSON.stringify(body);
  const current = state?.key === key ? state : null;

  async function runAudit() {
    const request = ++sequence.current;
    setState({ key, loading: true });
    try {
      const report = await auditDataset(body);
      if (sequence.current === request) setState({ key, loading: false, report });
    } catch (error) {
      if (sequence.current === request) {
        setState({ key, loading: false, error: error instanceof Error ? error.message : t.evidence.failed });
      }
    }
  }

  return (
    <section data-testid="dataset-audit" style={{ padding: 18, marginBottom: 18, border: "1px solid var(--vf-panel-stroke)", borderRadius: 12, display: "grid", gap: 12 }}>
      <strong>{t.evidence.title}</strong>
      <span style={{ fontSize: 12, color: "var(--vf-text-dim)" }}>{t.evidence.scope}</span>
      <TextField label={t.evidence.groupManifest} value={manifest} onChange={setManifest} placeholder={t.evidence.groupPlaceholder} help={t.evidence.groupHint} mono />
      <Toggle label={t.evidence.strictGroups} value={strict} onChange={setStrict} help={t.evidence.strictHint} />
      <button type="button" disabled={!baseDir.trim() || current?.loading} onClick={() => void runAudit()} aria-busy={current?.loading ?? false}
        style={{ justifySelf: "start", padding: "8px 14px", borderRadius: 8, border: "1px solid var(--accent-vf)", background: "var(--accent-soft)", color: "var(--vf-text)", cursor: current?.loading ? "wait" : "pointer", opacity: !baseDir.trim() || current?.loading ? 0.6 : 1 }}>
        {current?.loading ? t.evidence.auditing : t.evidence.audit}
      </button>
      {current?.error && <div role="alert">{t.evidence.failed} {current.error}</div>}
      {current?.report && <DatasetAuditResult report={current.report} />}
    </section>
  );
}
