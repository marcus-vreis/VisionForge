import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import { MLflowExportStatus } from "./MLflowExportStatus";
import { ProfileMenu } from "./ProfileMenu";
import { ProfileTrackingSettings } from "./ProfileTrackingSettings";

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };
const wrap = (element: React.ReactNode) => renderToStaticMarkup(<LanguageContext.Provider value={lang}>{element}</LanguageContext.Provider>);

describe("MLflow profile controls", () => {
  it("defaults legacy profiles to disabled and shows enabled metadata", () => {
    const profile = { slug: "default", name: "Default", is_default: true };
    const legacy = wrap(<ProfileTrackingSettings profile={profile} onSaved={() => {}} />);
    expect(legacy).toContain(pt.mlflow.label);
    expect(legacy).not.toContain('checked=""');
    const enabled = wrap(<ProfileTrackingSettings profile={{ ...profile, settings: { mlflow: { enabled: true } } }} onSaved={() => {}} />);
    expect(enabled).toContain('checked=""');
  });
  it("lives in the header's profile menu as a switch", () => {
    const profile = { slug: "default", name: "Marcus", is_default: true, settings: { mlflow: { enabled: true } } };
    const menu = wrap(<ProfileMenu name="Marcus" profile={profile} shared={false} initialOpen
      onProfileSaved={() => {}} onChangeName={() => {}} onSwitchProfile={() => {}} onNewProfile={() => {}} />);
    const input = menu.split(`${pt.mlflow.label}</span>`)[1]?.match(/^<input[^>]*>/)?.[0] ?? "";
    expect(input).toContain('role="switch"');
    expect(input).toContain('checked=""');
  });
  it("offers retry and the error when the export failed", () => {
    const html = wrap(<MLflowExportStatus runId="r1" tracking={{ state: "failed", error: "OSError" }} onUpdated={() => {}} />);
    expect(html).toContain(pt.mlflow.retry);
    expect(html).toContain(`${pt.mlflow.warning}: OSError`);
  });
  it("says how to install MLflow when the extra is missing", () => {
    const html = wrap(<MLflowExportStatus runId="r1" tracking={{ state: "disabled", error: "ModuleNotFoundError" }} onUpdated={() => {}} />);
    expect(html).toContain(pt.mlflow.retry);
    expect(html).toContain(pt.mlflow.notInstalled);
    expect(html).not.toContain("ModuleNotFoundError");
  });
  it("keeps old and successfully exported records free of warnings", () => {
    expect(wrap(<MLflowExportStatus runId="r1" onUpdated={() => {}} />)).toBe("");
    expect(wrap(<MLflowExportStatus runId="r1" tracking={{ state: "completed" }} onUpdated={() => {}} />)).toBe("");
  });
});
