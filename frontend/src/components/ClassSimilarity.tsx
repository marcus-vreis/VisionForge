import {
  classSimilaritySaved,
  classSimilarityStart,
  classSimilarityStatus,
  explorerScanCancel,
} from "../api/client";
import { useAnalysisJob } from "../lib/analysis-job";
import type { ClassSimilarityReport, ClassSimilarityRequest, SimilarityPair } from "../lib/class-similarity";
import { sharePct } from "../lib/class-similarity";
import { thumbUrl, type ExplorerForm } from "../lib/dataset-explorer";
import { useT } from "../i18n/useT";
import { ROUTE, useRouteOpen } from "../lib/session";
import { AnalysisJobCard } from "./AnalysisJobCard";
import { DataTable, LearnMore } from "./DatasetLabParts";
import { card, mono, text, warn } from "./datasetExplorerStyles";

/** Images described per class; the server caps it at 2000. */
const PER_CLASS = 200;

const API = {
  start: classSimilarityStart,
  status: classSimilarityStatus,
  saved: classSimilaritySaved,
  cancel: explorerScanCancel,
};

function Thumb({ root, path, caption }: { root: string; path: string; caption: string }) {
  return (
    <figure style={{ margin: 0, minWidth: 0 }}>
      <img
        src={thumbUrl(root, path, 128, null)}
        alt={path}
        loading="lazy"
        style={{ width: "100%", aspectRatio: "1", objectFit: "cover", borderRadius: 6 }}
      />
      <figcaption style={{ ...mono, fontSize: 11, wordBreak: "break-all" }} title={path}>
        {caption}
      </figcaption>
    </figure>
  );
}

function PairImages({ root, pair }: { root: string; pair: SimilarityPair }) {
  const t = useT().classSimilarity;
  return (
    <LearnMore summary={t.examplesTitle(pair.a, pair.b)}>
      <div style={{ display: "grid", gap: 10 }}>
        <div style={text}>{t.examplesHelp}</div>
        {pair.examples.map((e) => (
          <div
            key={e.path}
            style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 110px))", gap: 8 }}
            data-testid="similarity-example"
          >
            <Thumb root={root} path={e.path} caption={`${e.class} · ${e.path}`} />
            <Thumb root={root} path={e.neighbour} caption={t.nearest(e.neighbour_class, e.similarity)} />
          </div>
        ))}
      </div>
    </LearnMore>
  );
}

/** Which classes look alike, and the images where they meet (ADR-239). */
export function ClassSimilarityView({ report, root }: { report: ClassSimilarityReport; root: string }) {
  const t = useT().classSimilarity;
  const flagged = report.pairs.filter((p) => p.flagged);
  return (
    <div style={card} data-testid="class-similarity-report">
      <div style={text}>
        {t.described(report.described, report.per_class, report.model)}
        {report.sampled && ` ${t.sampled}`}
        {report.unreadable > 0 && ` ${t.unreadable(report.unreadable)}`}
      </div>
      {flagged.length > 0 ? (
        <div style={warn} role="status">
          {t.found(flagged.length, sharePct(report.flag_share))}
        </div>
      ) : (
        <div style={text}>{t.none(sharePct(report.flag_share))}</div>
      )}
      {report.pairs.length > 0 && (
        <DataTable>
          <thead>
            <tr>
              <th style={{ textAlign: "left" }}>{t.pairColumn}</th>
              <th style={{ textAlign: "right" }}>{t.aLooksLikeB}</th>
              <th style={{ textAlign: "right" }}>{t.bLooksLikeA}</th>
              <th style={{ textAlign: "right" }}>{t.centroid}</th>
            </tr>
          </thead>
          <tbody>
            {report.pairs.map((p) => (
              <tr key={`${p.a}|${p.b}`} style={p.flagged ? { color: "var(--vf-warn, #fbbf24)" } : undefined}>
                <td>
                  {p.a} · {p.b}
                </td>
                <td style={{ textAlign: "right" }}>{sharePct(p.a_to_b)}</td>
                <td style={{ textAlign: "right" }}>{sharePct(p.b_to_a)}</td>
                <td style={{ textAlign: "right" }}>{p.centroid_similarity.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </DataTable>
      )}
      {report.pair_count > report.pairs.length && (
        <div style={mono}>{t.morePairs(report.pair_count - report.pairs.length)}</div>
      )}
      {/* Every listed pair, flagged or not: one misfiled image is a small share. */}
      {report.pairs.map((p) => (
        <PairImages key={`${p.a}|${p.b}`} root={root} pair={p} />
      ))}
      {report.classes.length > 0 && (
        <LearnMore summary={t.purityTitle}>
          <div style={{ display: "grid", gap: 6 }}>
            <div style={text}>{t.purityHelp}</div>
            <div style={mono}>
              {report.classes.map((c) => (
                <div key={c.name}>
                  {c.name}: {c.purity === null ? "—" : sharePct(c.purity)} ({c.used})
                </div>
              ))}
            </div>
          </div>
        </LearnMore>
      )}
      <LearnMore summary={t.howTitle}>
        <div style={text}>{t.how}</div>
      </LearnMore>
    </div>
  );
}

function ClassSimilarityInner({ form }: { form: ExplorerForm }) {
  const t = useT().classSimilarity;
  const request: ClassSimilarityRequest = { ...form, per_class: PER_CLASS };
  const ready = form.base_dir.trim() !== "";
  const state = useAnalysisJob(API, request, ready);
  return (
    <section data-testid="class-similarity" style={{ display: "grid", gap: 12 }}>
      <AnalysisJobCard words={t} state={state} ready={ready} />
      {state.shown && <ClassSimilarityView report={state.shown} root={form.base_dir} />}
    </section>
  );
}

/** Classification only; absent where the server does not run it (ADR-216). */
export function ClassSimilarity({ form }: { form: ExplorerForm }) {
  const open = useRouteOpen(ROUTE.classSimilarity);
  return open && form.task === "classification" ? <ClassSimilarityInner form={form} /> : null;
}
