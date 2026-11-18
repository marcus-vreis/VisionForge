/**
 * Colab in the device selector and on the Dados step (ADR-211), rendered to
 * markup: what is offered, what is off and why. The requests are the browser
 * check's.
 */
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { ColabAccount } from "../hooks/useColabAccount";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import { defaultColabSetup, type ColabSetup } from "../lib/colab-target";
import { ColabTargetContext, type ColabTargetValue } from "../lib/colab-target-context";
import { ColabDatasetCard, ColabDeviceSection, type ColabDeviceChoice } from "./ColabTarget";

const c = pt.colab.target;
// The message names `<nome>`, which the markup escapes: its start is enough.
const badSource = pt.colab.badSource.split("<")[0];
const noop = () => {};

function markup(node: React.ReactNode) {
  return renderToStaticMarkup(
    <LanguageContext.Provider value={{ lang: "pt", t: pt, locale: "pt-BR", setLang: noop }}>
      {node}
    </LanguageContext.Provider>,
  );
}

const connected: ColabAccount = {
  stage: "connected",
  email: "ana@gmail.com",
  facts: { balance: "200", runtimes: 1, allocated: ["T4"] },
};
const choice: ColabDeviceChoice = { active: false, accelerator: "T4", onPick: noop };

function section(account: ColabAccount, over: Partial<ColabDeviceChoice> = {}) {
  return markup(<ColabDeviceSection account={account} choice={{ ...choice, ...over }} />);
}

describe("ColabDeviceSection", () => {
  it("offers one button per GPU and says what the account has", () => {
    const html = section(connected);
    for (const gpu of ["T4", "L4", "A100", "H100", "G4"]) {
      expect(html).toContain(`title="${c.option(gpu)}"`);
    }
    expect(html).toContain(c.account("ana@gmail.com"));
    expect(html).toContain(c.balance("200"));
    expect(html).toContain(c.runtimes(1));
    expect(html).toContain(c.allocated("T4"));
    expect(html).not.toContain("data-colab-target-off");
    expect(html).not.toContain('disabled=""');
  });

  it("marks the GPU in use only while Colab is the place to train", () => {
    expect(section(connected)).not.toContain('aria-pressed="true"');
    const html = section(connected, { active: true, accelerator: "L4" });
    expect(html.match(/aria-pressed="true"/g)).toHaveLength(1);
    expect(html).toMatch(/aria-pressed="true"[^>]*title="Colab · L4"/);
  });

  it("is off, saying where to connect, until the account is signed in", () => {
    const html = section({
      stage: "logged_out",
      email: null,
      facts: { balance: null, runtimes: 0, allocated: [] },
    });
    expect(html).toContain("data-colab-target-off");
    expect(html).toContain(c.notConnected);
    expect(html.match(/disabled=""/g)).toHaveLength(5);
  });

  it("is off with the reason when the open experiment cannot train there", () => {
    const html = section(connected, { blocked: c.onlyClassification });
    expect(html).toContain(c.onlyClassification);
    expect(html.match(/disabled=""/g)).toHaveLength(5);
  });
});

function card(over: Partial<ColabSetup>, value: Partial<ColabTargetValue> = {}) {
  const target: ColabTargetValue = {
    setup: { ...defaultColabSetup(), ...over },
    onChange: noop,
    formBaseDir: "",
    problem: null,
    ...value,
  };
  return markup(
    <ColabTargetContext.Provider value={target}>
      <ColabDatasetCard />
    </ColabTargetContext.Provider>,
  );
}

describe("ColabDatasetCard", () => {
  it("draws nothing while the training runs on this machine", () => {
    expect(markup(<ColabDatasetCard />)).toBe("");
  });

  it("asks for a My Drive folder first and warns about a backup-only folder", () => {
    const html = card({});
    expect(html).toContain('data-colab-dataset="drive"');
    expect(html).toContain(c.dataIntro);
    expect(html).toContain(c.kindDrive);
    expect(html).toContain(c.kindLink);
    expect(html).toContain("Computadores");
    expect(html).toContain(pt.colab.driveExport);
  });

  it("says the folder is missing when neither the Drive nor the form has one", () => {
    const html = card({}, { problem: "no_folder" });
    expect(html).toContain("data-colab-dataset-problem");
    expect(html).toContain(c.noFolder);
  });

  it("names the form's folder as the fallback when no Drive folder was picked", () => {
    const html = card({}, { formBaseDir: "G:\\Meu Drive\\beans" });
    expect(html).toContain("G:\\Meu Drive\\beans");
    expect(html).not.toContain("data-colab-dataset-problem");
  });

  it("takes a link instead, and refuses one the VM cannot download", () => {
    const empty = card({ kind: "link" });
    expect(empty).toContain('data-colab-dataset="link"');
    expect(empty).toContain(c.linkLabel);
    expect(empty).not.toContain(badSource);
    expect(empty).not.toContain("Computadores");
    expect(card({ kind: "link", link: "C:\\datasets\\beans" })).toContain(badSource);
    expect(card({ kind: "link", link: "hf:beans/beans" })).not.toContain(badSource);
  });
});
