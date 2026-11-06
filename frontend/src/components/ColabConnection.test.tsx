import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ColabStatus, ColabUsage } from "../api/client";
import { LanguageContext } from "../i18n/context";
import { pt } from "../i18n/pt";
import {
  cleanAuthCode,
  colabStage,
  parseDatasetSource,
  roundHours,
} from "../lib/colab-connection";
import { ColabPanel, type ColabPanelProps } from "./ColabConnection";

const lang = { lang: "pt" as const, t: pt, locale: "pt-BR", setLang: () => {} };

const installed: ColabStatus = {
  pinned_version: "0.7.5",
  installed_version: "0.7.5",
  install_state: "idle",
  install_error: null,
  logged_in: false,
  account: null,
  login_pending: false,
};

const noop = () => {};

function panel(over: Partial<ColabPanelProps>) {
  const props: ColabPanelProps = {
    status: installed,
    usage: null,
    usageError: false,
    loginUrl: null,
    error: null,
    busy: false,
    onInstall: noop,
    onLogin: noop,
    onFinish: noop,
    onCancel: noop,
    onSwitch: noop,
    sessions: null,
    onStopSession: noop,
    ...over,
  };
  return renderToStaticMarkup(
    <LanguageContext.Provider value={lang}>
      <ColabPanel {...props} />
    </LanguageContext.Provider>,
  );
}

describe("colabStage", () => {
  it("is not_installed until the pinned version is there", () => {
    expect(colabStage({ ...installed, installed_version: null })).toBe("not_installed");
    // Another version (an old pin) counts as not installed: the install fixes it.
    expect(colabStage({ ...installed, installed_version: "0.7.4" })).toBe("not_installed");
  });

  it("reports the install while it runs and when it failed", () => {
    expect(colabStage({ ...installed, installed_version: null, install_state: "installing" })).toBe(
      "installing",
    );
    expect(colabStage({ ...installed, installed_version: null, install_state: "failed" })).toBe(
      "install_failed",
    );
  });

  it("follows the login", () => {
    expect(colabStage(installed)).toBe("logged_out");
    expect(colabStage({ ...installed, login_pending: true })).toBe("login_pending");
    expect(colabStage({ ...installed, logged_in: true })).toBe("connected");
  });
});

describe("cleanAuthCode", () => {
  it("trims the pasted code and refuses blanks and inner spaces", () => {
    expect(cleanAuthCode("  4/0AbC-dEf \n")).toBe("4/0AbC-dEf");
    expect(cleanAuthCode("   ")).toBeNull();
    expect(cleanAuthCode("4/ab cd")).toBeNull();
    expect(cleanAuthCode("x".repeat(513))).toBeNull();
  });
});

describe("roundHours", () => {
  it("keeps one decimal below 10 h and whole hours above", () => {
    expect(roundHours(332.11)).toBe(332);
    expect(roundHours(3.27)).toBe(3.3);
  });
});

describe("ColabPanel", () => {
  it("offers the pinned install when the CLI is missing", () => {
    const html = panel({ status: { ...installed, installed_version: null } });
    expect(html).toContain('data-colab-stage="not_installed"');
    expect(html).toContain(pt.colab.install("0.7.5"));
  });

  it("shows why the install failed and offers a retry", () => {
    const html = panel({
      status: { ...installed, installed_version: null, install_state: "failed", install_error: "sem rede" },
    });
    expect(html).toContain("sem rede");
    expect(html).toContain(pt.colab.retry);
  });

  it("starts with a sign-in button, then shows the URL and the code field", () => {
    expect(panel({})).toContain(pt.colab.login);
    const html = panel({
      status: { ...installed, login_pending: true },
      loginUrl: "https://accounts.google.com/o/oauth2/auth?x=1",
    });
    expect(html).toContain('href="https://accounts.google.com/o/oauth2/auth?x=1"');
    expect(html).toContain('id="colab-auth-code"');
  });

  it("shows the account, the balance and the hours left when connected", () => {
    const usage: ColabUsage = { balance: 381.93, rate_per_hour: 1.15, active_assignments: 1, hours_left: 332.11 };
    const html = panel({ status: { ...installed, logged_in: true, account: "pessoa@example.com" }, usage });
    expect(html).toContain(pt.colab.account("pessoa@example.com"));
    expect(html).toContain(pt.colab.balance("381.93"));
    expect(html).toContain(pt.colab.hoursLeft(332, "1.15"));
    expect(html).toContain(pt.colab.switchAccount);
  });

  it("says the hours depend on the GPU while no runtime is active", () => {
    const usage: ColabUsage = { balance: 381.93, rate_per_hour: 0, active_assignments: 0, hours_left: null };
    const html = panel({ status: { ...installed, logged_in: true, account: "a@b.c" }, usage });
    expect(html).toContain(pt.colab.noRate);
  });
});

describe("parseDatasetSource", () => {
  it("reads a zip URL and the two public providers", () => {
    expect(parseDatasetSource(" https://x.org/coffee.zip ")).toEqual({ kind: "url", url: "https://x.org/coffee.zip" });
    expect(parseDatasetSource("torchvision:cifar10")).toEqual({
      kind: "provider",
      provider: "torchvision",
      dataset: "cifar10",
    });
    expect(parseDatasetSource("hf:owner/beans")).toEqual({
      kind: "provider",
      provider: "huggingface",
      dataset: "owner/beans",
    });
  });

  it("refuses a local path or a file URL: the VM cannot read this machine", () => {
    expect(parseDatasetSource("C:/datasets/coffee")).toBeNull();
    expect(parseDatasetSource("file:///C:/datasets/coffee.zip")).toBeNull();
    expect(parseDatasetSource("roboflow:ws/proj")).toBeNull();
    expect(parseDatasetSource("")).toBeNull();
  });
});

describe("ColabPanel when connected", () => {
  const connected = { ...installed, logged_in: true, account: "a@b.c" };

  it("offers training only when the open task can train on Colab", () => {
    expect(panel({ status: connected })).not.toContain("data-colab-train");
    const html = panel({ status: connected, onTrain: noop });
    expect(html).toContain("data-colab-train");
    expect(html).toContain(pt.colab.train);
    expect(html).toContain('<option value="T4" selected="">T4</option>');
  });

  it("lists the runtimes alive and stops only the ones VisionForge started", () => {
    const html = panel({
      status: connected,
      sessions: [
        { name: "vf-20261105-140000", hardware: "T4", ours: true },
        { name: "?", hardware: "CPU", ours: false },
      ],
    });
    expect(html).toContain("vf-20261105-140000 · T4");
    expect(html.match(new RegExp(pt.colab.stopSession, "g"))?.length).toBe(1);
    expect(html).toContain(pt.colab.sessionNotOurs);
  });

  it("says when no runtime is alive", () => {
    expect(panel({ status: connected, sessions: [] })).toContain(pt.colab.sessionsNone);
  });
});
