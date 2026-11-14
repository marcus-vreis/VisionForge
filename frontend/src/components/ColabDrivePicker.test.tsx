import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ColabDriveFolders } from "../api/client";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import { DrivePickerView, type DrivePickerViewProps, type DriveView } from "./ColabDrivePicker";

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };
const noop = () => {};

const ROOT: ColabDriveFolders = {
  root: "G:\\Meu Drive",
  rel: "",
  path: "G:\\Meu Drive",
  remote: "/content/drive/MyDrive",
  folders: ["alfa", "datasets"],
  truncated: false,
  can_open: true,
};

const NO_DRIVE: ColabDriveFolders = {
  root: null,
  rel: "",
  path: null,
  remote: null,
  folders: [],
  truncated: false,
  can_open: false,
};

function picker(view: Partial<DriveView>, over: Partial<DrivePickerViewProps> = {}) {
  const props: DrivePickerViewProps = {
    view: { listing: null, loading: false, error: null, ...view },
    browsing: false,
    folder: "",
    remote: null,
    driveRoot: "",
    onFolder: noop,
    onDriveRoot: noop,
    onRetry: noop,
    onBrowse: noop,
    onEnter: noop,
    onUse: noop,
    onCancelBrowse: noop,
    onOpen: noop,
    ...over,
  };
  return renderToStaticMarkup(
    <LanguageContext.Provider value={lang}>
      <DrivePickerView {...props} />
    </LanguageContext.Provider>,
  );
}

describe("DrivePickerView", () => {
  it("says it is looking for Drive for desktop first", () => {
    const html = picker({ loading: true });
    expect(html).toContain('data-colab-drive="checking"');
    expect(html).toContain(pt.colab.driveChecking);
    expect(html).not.toContain(pt.colab.drivePick);
  });

  it("explains Drive for desktop in three steps when no My Drive is found", () => {
    const html = picker({ listing: NO_DRIVE });
    expect(html).toContain('data-colab-drive="missing"');
    for (const step of [pt.colab.driveStep1, pt.colab.driveStep2, pt.colab.driveStep3]) {
      expect(html).toContain(step.replaceAll('"', "&quot;"));
    }
    // The typed root stays as the fallback; there is nothing to browse or open.
    expect(html).toContain('id="colab-drive-root"');
    expect(html).toContain(pt.colab.driveRetry);
    expect(html).not.toContain("data-colab-drive-browser");
    expect(html).not.toContain(pt.colab.drivePick);
    expect(html).not.toContain(pt.colab.driveOpen);
  });

  it("shows the server's error and keeps the typed root when the lookup failed", () => {
    const html = picker({ error: "sem servidor" });
    expect(html).toContain('data-colab-drive="failed"');
    expect(html).toContain("sem servidor");
    expect(html).toContain('id="colab-drive-root"');
  });

  it("offers the folder field, the browser and the file manager once My Drive is found", () => {
    const html = picker({ listing: ROOT });
    expect(html).toContain('data-colab-drive="found"');
    expect(html).toContain('id="colab-drive-folder"');
    expect(html).toContain(pt.colab.drivePick);
    expect(html).toContain(pt.colab.driveOpen);
    expect(html).not.toContain("data-colab-drive-browser");
    // The typed root is still reachable, behind a closed disclosure.
    expect(html).toMatch(/<details><summary[^>]*>[^<]*<\/summary>.*id="colab-drive-root"/);
  });

  it("hides the file manager button where the server cannot open one", () => {
    expect(picker({ listing: { ...ROOT, can_open: false } })).not.toContain(pt.colab.driveOpen);
  });

  it("fills the field with the picked folder and shows its Colab path under it", () => {
    const html = picker(
      { listing: ROOT },
      { folder: "G:\\Meu Drive\\datasets\\flores", remote: "/content/drive/MyDrive/datasets/flores" },
    );
    expect(html).toContain('value="G:\\Meu Drive\\datasets\\flores"');
    expect(html).toContain(pt.colab.driveRemote("/content/drive/MyDrive/datasets/flores"));
  });

  it("browses with a breadcrumb from My Drive and a button per folder", () => {
    const listing: ColabDriveFolders = {
      ...ROOT,
      rel: "datasets/flores",
      path: "G:\\Meu Drive\\datasets\\flores",
      remote: "/content/drive/MyDrive/datasets/flores",
      folders: ["train", "val"],
    };
    const html = picker({ listing }, { browsing: true });
    expect(html).toContain("data-colab-drive-browser");
    const crumbs = [pt.colab.driveHome, "datasets", "flores"].map((name) => html.indexOf(`>${name}</button>`));
    expect(crumbs.every((i) => i >= 0)).toBe(true);
    expect([...crumbs].sort((a, b) => a - b)).toEqual(crumbs);
    expect(html).toMatch(/aria-current="location"[^>]*>flores<\/button>/);
    expect(html.match(/data-drive-folder/g)?.length).toBe(2);
    expect(html).toContain(">train</span>");
    expect(html).toContain(pt.colab.driveUse);
    expect(html).toContain(pt.colab.driveRemote("/content/drive/MyDrive/datasets/flores"));
  });

  it("says when a folder has no sub-folders, and when the list was cut", () => {
    expect(picker({ listing: { ...ROOT, folders: [] } }, { browsing: true })).toContain(pt.colab.driveEmpty);
    const cut = picker({ listing: { ...ROOT, truncated: true } }, { browsing: true });
    expect(cut).toContain(pt.colab.driveTruncated(2));
  });

  it("marks the browser busy while a folder loads and shows a failed one", () => {
    const busy = picker({ listing: ROOT, loading: true }, { browsing: true });
    expect(busy).toContain(pt.colab.driveLoading);
    expect(busy).toMatch(/<button[^>]*data-drive-use="true"[^>]*disabled=""/);
    const failed = picker({ listing: ROOT, error: "A pasta sumiu" }, { browsing: true });
    expect(failed).toContain('role="alert"');
    expect(failed).toContain("A pasta sumiu");
  });
});
