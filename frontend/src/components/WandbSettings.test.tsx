import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { WandbExportStatus as Status } from "../api/client";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import { WandbExportStatus } from "./WandbExportStatus";
import { WandbProfileSettings } from "./WandbProfileSettings";

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };
const wrap = (element: React.ReactNode) =>
  renderToStaticMarkup(<LanguageContext.Provider value={lang}>{element}</LanguageContext.Provider>);
const status = (fields: Partial<Status>): Status => ({ state: "failed", status: "FINISHED", ...fields });
const profile = { slug: "ana", name: "Ana", is_default: false };
// Static markup escapes the "&" of "W&B".
const html = (text: string) => text.replaceAll("&", "&amp;");

describe("W&B profile settings", () => {
  it("shows only the switch while it is off, also for an old profile", () => {
    const markup = wrap(<WandbProfileSettings profile={profile} onSaved={() => {}} />);
    expect(markup).toContain(html(pt.wandb.label));
    expect(markup).not.toContain('checked=""');
    expect(markup).not.toContain(html(pt.wandb.token));
  });
  it("asks for the destination and the key once it is on", () => {
    const settings = {
      mlflow: { enabled: false },
      wandb: { enabled: true, project: "lab-runs", entity: "lab" },
    };
    const markup = wrap(<WandbProfileSettings profile={{ ...profile, settings }} onSaved={() => {}} />);
    expect(markup).toContain('checked=""');
    expect(markup).toContain('value="lab-runs"');
    expect(markup).toContain('value="lab"');
    expect(markup).toContain('type="password"');
    expect(markup).toContain(html(pt.wandb.saveToken));
  });
});

describe("W&B run status", () => {
  it("links to the run once it is there", () => {
    const markup = wrap(<WandbExportStatus runId="r1" onUpdated={() => {}}
      status={status({ state: "finished", url: "https://wandb.ai/ana/visionforge/runs/abc" })} />);
    expect(markup).toContain('href="https://wandb.ai/ana/visionforge/runs/abc"');
    expect(markup).toContain('rel="noopener noreferrer"');
    expect(markup).toContain(html(pt.wandb.open));
  });
  it("never turns a non-https address into a link", () => {
    const markup = wrap(<WandbExportStatus runId="r1" onUpdated={() => {}}
      status={status({ state: "finished", url: "javascript:alert(1)" })} />);
    expect(markup).toBe("");
  });
  it.each([
    [{ state: "disabled" as const, error: "ModuleNotFoundError" }, pt.wandb.notInstalled],
    [{ error: "WandbTokenMissingError" }, pt.wandb.tokenMissing],
    [{ error: "WandbPrivateProjectError" }, pt.wandb.notPrivate],
    [{ error: "WandbRequestError" }, `${pt.wandb.warning}: WandbRequestError`],
  ])("explains %o and offers a retry", (fields, text) => {
    const markup = wrap(<WandbExportStatus runId="r1" onUpdated={() => {}} status={status(fields)} />);
    expect(markup).toContain(html(text));
    expect(markup).toContain(html(pt.wandb.retry));
  });
  it("says nothing for a run without W&B or still running", () => {
    expect(wrap(<WandbExportStatus runId="r1" onUpdated={() => {}} />)).toBe("");
    expect(wrap(<WandbExportStatus runId="r1" onUpdated={() => {}}
      status={status({ state: "running", status: "RUNNING" })} />)).toBe("");
  });
});
