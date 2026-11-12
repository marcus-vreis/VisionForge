import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { CometExportStatus as Status } from "../api/client";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import { CometExportStatus } from "./CometExportStatus";
import { CometProfileSettings } from "./CometProfileSettings";

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };
const wrap = (element: React.ReactNode) =>
  renderToStaticMarkup(<LanguageContext.Provider value={lang}>{element}</LanguageContext.Provider>);
const status = (fields: Partial<Status>): Status => ({ state: "failed", status: "FINISHED", ...fields });
const profile = { slug: "ana", name: "Ana", is_default: false };

describe("Comet profile settings", () => {
  it("shows only the switch while it is off, also for a profile saved before Comet", () => {
    const settings = {
      mlflow: { enabled: false },
      wandb: { enabled: true, project: "lab-runs", entity: "" },
    };
    const markup = wrap(<CometProfileSettings profile={{ ...profile, settings }} onSaved={() => {}} />);
    expect(markup).toContain(pt.comet.label);
    expect(markup).not.toContain('checked=""');
    expect(markup).not.toContain(pt.comet.token);
  });
  it("asks for the destination and the key once it is on", () => {
    const settings = {
      mlflow: { enabled: false },
      comet: { enabled: true, project: "lab-runs", workspace: "lab" },
    };
    const markup = wrap(<CometProfileSettings profile={{ ...profile, settings }} onSaved={() => {}} />);
    expect(markup).toContain('checked=""');
    expect(markup).toContain('value="lab-runs"');
    expect(markup).toContain('value="lab"');
    expect(markup).toContain('type="password"');
    expect(markup).toContain(pt.comet.saveToken);
  });
});

describe("Comet run status", () => {
  it("links to the experiment once it is there", () => {
    const markup = wrap(<CometExportStatus runId="r1" onUpdated={() => {}}
      status={status({ state: "finished", url: "https://www.comet.com/ana/visionforge/abc" })} />);
    expect(markup).toContain('href="https://www.comet.com/ana/visionforge/abc"');
    expect(markup).toContain('rel="noopener noreferrer"');
    expect(markup).toContain(pt.comet.open);
  });
  it("never turns a non-https address into a link", () => {
    const markup = wrap(<CometExportStatus runId="r1" onUpdated={() => {}}
      status={status({ state: "finished", url: "javascript:alert(1)" })} />);
    expect(markup).toBe("");
  });
  it.each([
    [{ state: "disabled" as const, error: "ModuleNotFoundError" }, pt.comet.notInstalled],
    [{ error: "CometTokenMissingError" }, pt.comet.tokenMissing],
    [{ error: "CometPrivateProjectError" }, pt.comet.notPrivate],
    [{ error: "CometBusyError" }, pt.comet.busy],
    [{ error: "CometRequestError" }, `${pt.comet.warning}: CometRequestError`],
  ])("explains %o and offers a retry", (fields, text) => {
    const markup = wrap(<CometExportStatus runId="r1" onUpdated={() => {}} status={status(fields)} />);
    expect(markup).toContain(text);
    expect(markup).toContain(pt.comet.retry);
  });
  it("says nothing for a run without Comet or still running", () => {
    expect(wrap(<CometExportStatus runId="r1" onUpdated={() => {}} />)).toBe("");
    expect(wrap(<CometExportStatus runId="r1" onUpdated={() => {}}
      status={status({ state: "running", status: "RUNNING" })} />)).toBe("");
  });
});
