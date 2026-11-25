import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it } from "vitest";

import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { LOCAL_SESSION, setSession, type Session } from "../lib/session";
import { OwnDatasets } from "./OwnDatasets";
import { OwnDatasetUploadView, type UploadPhase } from "./OwnDatasetUpload";

const MIB = 1024 ** 2;
const SERVER: Session = {
  ...LOCAL_SESSION,
  server_mode: true,
  user: "ana@lab.org",
  profile: "ana",
};
const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };
const noop = () => {};

function view(
  fields: Partial<Parameters<typeof OwnDatasetUploadView>[0]> & { phase?: UploadPhase } = {},
): string {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={lang}>
      <OwnDatasetUploadView
        phase="idle"
        name=""
        sent={0}
        total={0}
        error={null}
        notice={null}
        onName={noop}
        deleteAfterRun={false}
        onDeleteAfterRun={noop}
        onPick={noop}
        onResume={noop}
        onCancel={noop}
        {...fields}
      />
    </LanguageContext.Provider>,
  );
}

afterEach(() => setSession(LOCAL_SESSION));

describe("sending a dataset .zip to a shared server (ADR-246)", () => {
  it("offers the folder name and the button while nothing is on its way", () => {
    const markup = view();
    expect(markup).toContain(pt.ownDatasets.uploadName);
    expect(markup).toContain(pt.ownDatasets.uploadNameHint);
    expect(markup).toContain(pt.ownDatasets.upload);
    expect(markup).not.toMatch(/<button[^>]*disabled/);
    expect(markup).not.toContain('role="progressbar"');
    expect(markup).not.toContain(pt.ownDatasets.uploadResume);
  });

  it("shows how much went up while sending, with the button held", () => {
    const markup = view({ phase: "sending", sent: 3 * MIB, total: 12 * MIB });
    expect(markup).toContain('aria-valuenow="25"');
    expect(markup).toContain(pt.ownDatasets.uploadProgress(25, "3,0 MB", "12,0 MB"));
    expect(markup.match(/<button[^>]*disabled/g)).toHaveLength(1);
  });

  it("says the server is unpacking once every part is there", () => {
    const markup = view({ phase: "finishing", sent: 12 * MIB, total: 12 * MIB });
    expect(markup).toContain(pt.ownDatasets.uploadFinishing);
    expect(markup).toContain('aria-valuenow="100"');
  });

  it("offers to resume or cancel after a failure, with the server's reason", () => {
    const reason = "Ainda faltam partes deste envio (2).";
    const markup = view({ phase: "failed", error: reason, sent: MIB, total: 3 * MIB });
    expect(markup).toContain(reason);
    expect(markup).toContain('role="alert"');
    expect(markup).toContain(pt.ownDatasets.uploadResume);
    expect(markup).toContain(pt.ownDatasets.uploadCancel);
    // A new file waits until this one is resumed or cancelled.
    expect(markup.match(/<button[^>]*disabled/g)).toHaveLength(1);
  });

  it("offers to delete the dataset after the next training, off by default (ADR-249)", () => {
    const off = view();
    expect(off).toContain(pt.ownDatasets.deleteAfterRun);
    expect(off).toMatch(/<input type="checkbox"(?![^>]*checked)[^>]*>/);
    expect(off).not.toContain(pt.ownDatasets.deleteAfterRunHint);

    const on = view({ deleteAfterRun: true });
    expect(on).toMatch(/<input type="checkbox"[^>]*checked=""[^>]*>/);
    expect(on).toContain(pt.ownDatasets.deleteAfterRunHint);
    expect(on).not.toMatch(/<input type="checkbox"[^>]*disabled/);
  });

  it("holds the choice while the file is on its way", () => {
    const markup = view({ phase: "sending", deleteAfterRun: true, sent: MIB, total: 4 * MIB });
    expect(markup).toMatch(/<input type="checkbox"[^>]*disabled=""[^>]*>/);
  });

  it("says when the dataset arrived", () => {
    expect(view({ notice: pt.ownDatasets.uploaded("beans") })).toContain(
      pt.ownDatasets.uploaded("beans"),
    );
  });

  it("speaks English too", () => {
    const markup = renderToStaticMarkup(
      <LanguageContext.Provider value={{ ...lang, lang: "en", t: en, locale: "en-US" }}>
        <OwnDatasetUploadView
          phase="sending"
          name=""
          sent={MIB}
          total={4 * MIB}
          error={null}
          notice={null}
          onName={noop}
          deleteAfterRun={false}
          onDeleteAfterRun={noop}
          onPick={noop}
          onResume={noop}
          onCancel={noop}
        />
      </LanguageContext.Provider>,
    );
    expect(markup).toContain(en.ownDatasets.uploadProgress(25, "1.0 MB", "4.0 MB"));
  });
});

describe("my datasets", () => {
  const mine = () =>
    renderToStaticMarkup(
      <LanguageContext.Provider value={lang}>
        <OwnDatasets />
      </LanguageContext.Provider>,
    );

  it("offers 'send .zip' only where the server takes uploads", () => {
    setSession(SERVER);
    expect(mine()).toContain(pt.ownDatasets.upload);
    setSession({ ...SERVER, closed: ["POST /api/uploads"] });
    expect(mine()).not.toContain(pt.ownDatasets.upload);
  });
});
