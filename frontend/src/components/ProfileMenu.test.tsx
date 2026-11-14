import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { LanguageContext } from "../i18n/context";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import { DEFAULT_PROFILE_INFO, type ProfileInfo } from "../lib/profile";
import { ProfileMenu } from "./ProfileMenu";
import { WelcomeOverlay } from "./WelcomeOverlay";

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };
const wrap = (element: React.ReactNode) =>
  renderToStaticMarkup(<LanguageContext.Provider value={lang}>{element}</LanguageContext.Provider>);
// Static markup escapes the "&" of "W&B".
const html = (text: string) => text.replaceAll("&", "&amp;");
const noop = () => {};
const marcus: ProfileInfo = { ...DEFAULT_PROFILE_INFO, name: "Marcus" };

function menu(fields: { profile?: ProfileInfo | null; shared?: boolean; open?: boolean } = {}) {
  return wrap(
    <ProfileMenu
      name="Marcus"
      profile={fields.profile === undefined ? marcus : fields.profile}
      shared={fields.shared ?? false}
      onProfileSaved={noop}
      onChangeName={noop}
      onSwitchProfile={noop}
      onNewProfile={noop}
      initialOpen={fields.open ?? true}
    />,
  );
}

describe("the header's profile menu", () => {
  it("is one chip with the profile's name, closed until clicked", () => {
    const markup = menu({ open: false });
    expect(markup).toContain("Marcus");
    expect(markup).toContain('aria-haspopup="dialog"');
    expect(markup).toContain('aria-expanded="false"');
    expect(markup).not.toContain('role="dialog"');
  });

  it("holds one switch per tracker, each with its brand mark", () => {
    const markup = menu();
    expect(markup).toContain('role="dialog"');
    expect(markup).toContain(pt.profile.integrations);
    expect(markup.match(/role="switch"/g)).toHaveLength(3);
    for (const label of [pt.mlflow.label, pt.wandb.label, pt.comet.label]) {
      expect(markup).toContain(html(label));
    }
    // simple-icons' MLflow and W&B marks, and lucide's Orbit for Comet.
    expect(markup).toContain("M11.883.002a12.044");
    expect(markup).toContain("M2.48 0a1.55");
    expect(markup).toContain("lucide-orbit");
  });

  it("shows a tracker's fields only while it is on", () => {
    const off = menu();
    expect(off).not.toContain('checked=""');
    expect(off).not.toContain('type="password"');
    const on = menu({
      profile: {
        ...marcus,
        settings: {
          mlflow: { enabled: true },
          wandb: { enabled: true, project: "lab-runs", entity: "lab" },
          comet: { enabled: false, project: "visionforge", workspace: "" },
        },
      },
    });
    expect(on.match(/checked=""/g)).toHaveLength(2);
    expect(on).toContain('value="lab-runs"');
    expect(on).toContain(html(pt.wandb.token));
    expect(on).not.toContain(pt.comet.token);
  });

  it("offers a new profile with one, and switching with more than one", () => {
    const single = menu({ shared: false });
    expect(single).toContain(pt.header.changeName);
    expect(single).toContain(pt.profile.createTitle);
    expect(single).not.toContain(pt.profile.switchProfile);
    const many = menu({ shared: true });
    expect(many).toContain(pt.profile.switchProfile);
    expect(many).not.toContain(pt.profile.createTitle);
  });

  it("has no integrations before the profile is known", () => {
    const markup = menu({ profile: null });
    expect(markup).toContain(pt.header.changeName);
    expect(markup).not.toContain('role="switch"');
  });
});

describe("the profile picker", () => {
  it("has no tracking controls and no 'default' wording", () => {
    const markup = wrap(<WelcomeOverlay onName={noop} />);
    expect(markup).toContain(pt.profile.who);
    expect(markup).not.toContain(html(pt.mlflow.label));
    expect(markup).not.toContain(html(pt.wandb.label));
    expect(markup).not.toContain(pt.comet.label);
    expect(markup).not.toContain('type="checkbox"');
    // The default profile still without a name is listed as unnamed.
    expect(markup).toContain(pt.profile.unnamed);
    expect(markup.toLowerCase()).not.toContain("padrão");
  });

  it("keeps the name step to the name and the pronoun", () => {
    const markup = wrap(<WelcomeOverlay onName={noop} />);
    // Only the picker's own link creates a profile; the name step has none.
    expect(markup.split(pt.profile.create)).toHaveLength(2);
    expect(markup).toContain(pt.welcome.askName);
    expect(markup).toContain(pt.welcome.pronounLabel);
  });

  it("names nothing 'default' in either language", () => {
    for (const dict of [pt.profile, en.profile]) {
      const text = Object.values(dict)
        .map((v) => (typeof v === "function" ? String(v("x")) : String(v)))
        .join(" ")
        .toLowerCase();
      expect(text).not.toMatch(/padrão|default/);
    }
  });
});
