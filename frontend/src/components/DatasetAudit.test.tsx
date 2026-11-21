import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { DatasetAuditReport } from "../api/client";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { auditRequest } from "../lib/dataset-audit";
import { DatasetAudit, DatasetAuditResult } from "./DatasetAudit";
import { DownloadBundleButton } from "./DownloadBundleButton";

const clean: DatasetAuditReport = {
  ok: true, root: "/data", hash_algorithm: "sha256", visual_similarity_checked: false,
  image_counts: { train: 2, val: 1, test: 1 }, duplicate_groups: [], missing_splits: [],
  empty_splits: [], unreadable_count: 0, unreadable: [], group_manifest: null,
  strict_groups: false, group_overlaps: [], near_duplicate_groups: [], near_duplicate_max_distance: 6,
};

function render(lang: "pt" | "en", node: React.ReactNode) {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={{ lang, t: lang === "pt" ? pt : en, locale: lang, setLang: () => {} }}>
      {node}
    </LanguageContext.Provider>,
  );
}

describe("dataset audit evidence", () => {
  it.each(["pt", "en"] as const)("shows measured counts and the visual-similarity limit in %s", (lang) => {
    const t = lang === "pt" ? pt : en;
    const html = render(lang, <DatasetAuditResult report={clean} />);
    expect(html).toContain(t.evidence.clean);
    expect(html).toContain(t.evidence.scope);
    expect(html).toContain(t.evidence.imageCount("train", 2));
    expect(html).toContain(t.evidence.duplicateCount(0));
    expect(html).toContain(t.evidence.groupCount(0));
    expect(html).not.toContain("<details>");
  });

  it("reports every finding category, bounds details and escapes file names", () => {
    const report = { ...clean, ok: false,
      duplicate_groups: Array.from({ length: 7 }, (_, i) => ({ sha256: `hash${i}`, splits: ["train", "test"], paths: ["<unsafe>.png", ...Array.from({ length: 8 }, (_, n) => `path-${n}.png`)] })),
      missing_splits: ["val"], empty_splits: ["test"], unreadable_count: 1,
      unreadable: [{ path: "broken.png", reason: "invalid header" }],
      group_overlaps: [{ group: "patient1", splits: ["train", "test"], paths: ["a.png", "b.png"] }],
    };
    const html = render("en", <DatasetAuditResult report={report} />);
    for (const text of [en.evidence.findings, en.evidence.duplicateCount(7), en.evidence.missingCount(1), en.evidence.emptyCount(1), en.evidence.unreadableCount(1), en.evidence.groupCount(1), "patient1", "broken.png", "&lt;unsafe&gt;.png"]) expect(html).toContain(text);
    expect(html).toContain("hash4");
    expect(html).not.toContain("hash5");
    expect(html).not.toContain("path-4.png");
  });

  it.each(["pt", "en"] as const)("shows the near-duplicates found across splits in %s", (lang) => {
    const t = lang === "pt" ? pt : en;
    const report = { ...clean, ok: false, visual_similarity_checked: true,
      near_duplicate_groups: [{ splits: ["test", "train"], paths: ["test/cat/a.jpg", "train/cat/a.png"] }],
    };
    const html = render(lang, <DatasetAuditResult report={report} />);
    expect(html).toContain(t.evidence.scopeNear(6));
    expect(html).not.toContain(t.evidence.scope);
    expect(html).toContain(t.evidence.nearCount(1));
    expect(html).toContain("test/cat/a.jpg");
    expect(html).toContain("train/cat/a.png");
  });

  it("says nothing about near-duplicates when they were not looked for", () => {
    const html = render("pt", <DatasetAuditResult report={clean} />);
    expect(html).not.toContain(pt.evidence.nearCount(0));
  });

  it("sends a regression CSV layout with its columns and drops a blank split", () => {
    const body = auditRequest(
      { baseDir: "/d", splits: { train: "fit.csv", val: "check.csv", test: " " }, layout: "csv", imagesDir: "pics", imageColumn: "", optionalSplits: ["test"] },
      { near_duplicates: true },
    );
    expect(body).toEqual({
      base_dir: "/d",
      splits: { train: "fit.csv", val: "check.csv" },
      layout: "csv",
      images_dir: "pics",
      image_column: "image",
      optional_splits: ["test"],
      near_duplicates: true,
    });
  });

  it("sends no CSV fields with a folder or YOLO layout", () => {
    const body = auditRequest({ baseDir: "/d", splits: { train: "train/images" }, imagesDir: "pics" }, {});
    expect(body).toEqual({ base_dir: "/d", splits: { train: "train/images" }, layout: "folders" });
    expect(auditRequest({ baseDir: "/d", splits: { train: "train" }, layout: "yolo" }, {}).layout).toBe("yolo");
  });

  it("shows an absent optional split as no finding", () => {
    const report = { ...clean, image_counts: { train: 2, val: 1 }, absent_optional_splits: ["test"] };
    const html = render("pt", <DatasetAuditResult report={report} />);
    expect(html).toContain(pt.evidence.optionalAbsent("test"));
    expect(html).toContain(pt.evidence.clean);
  });

  it("states what the task's audit reads", () => {
    const html = render("pt", <DatasetAudit baseDir="/d" splits={{ train: "train" }} note={pt.evidence.notes.masks} />);
    expect(html).toContain(pt.evidence.notes.masks);
  });

  it("starts idle, requires a dataset and exposes optional group controls", () => {
    const html = render("pt", <DatasetAudit baseDir="" splits={{ train: "fit", val: "valid", test: "final" }} />);
    expect(html).toContain(pt.evidence.audit);
    expect(html).toContain(pt.evidence.groupManifest);
    expect(html).toContain(pt.evidence.strictGroups);
    expect(html).toContain(pt.evidence.nearDuplicates);
    expect(html).toContain('disabled=""');
    expect(html).not.toContain('data-testid="audit-result"');
    expect(html).not.toContain('role="alert"');
  });

  it.each(["pt", "en"] as const)("localizes the bundle action in %s", (lang) => {
    const t = lang === "pt" ? pt : en;
    const html = render(lang, <DownloadBundleButton runId="r" />);
    expect(html).toContain(t.evidence.bundle);
    expect(html).toContain('aria-busy="false"');
    expect(html).not.toContain('role="alert"');
  });
});
