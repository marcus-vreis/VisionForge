import { afterEach, describe, expect, it, vi } from "vitest";
import { en } from "../i18n/en";
import { pt } from "../i18n/pt";
import {
  captureGitHubOAuthResult,
  clearGitHubOAuthResult,
  gitHubOAuthNotice,
  isGitHubAuthorizeUrl,
  openGitHubAuthorization,
  peekGitHubOAuthResult,
} from "./github-oauth";

const AUTHORIZE = "https://github.com/login/oauth/authorize?client_id=x&state=y";

function history() {
  return { state: { keep: 1 }, replaceState: vi.fn() };
}

afterEach(() => clearGitHubOAuthResult());

describe("the way back from GitHub (ADR-250)", () => {
  it.each(["connected", "denied", "failed"] as const)("keeps %s and cleans the address", (result) => {
    const h = history();
    const href = `http://localhost:8000/?github_oauth=${result}&x=1#top`;
    expect(captureGitHubOAuthResult({ href }, h)).toBe(result);
    expect(peekGitHubOAuthResult()).toBe(result);
    expect(h.replaceState).toHaveBeenCalledWith({ keep: 1 }, "", "/?x=1#top");
    clearGitHubOAuthResult();
    expect(peekGitHubOAuthResult()).toBeNull();
  });

  it("leaves an address without the flag alone", () => {
    const h = history();
    expect(captureGitHubOAuthResult({ href: "http://localhost:8000/?x=1" }, h)).toBeNull();
    expect(h.replaceState).not.toHaveBeenCalled();
  });

  it("cleans an unknown value without showing anything", () => {
    const h = history();
    expect(captureGitHubOAuthResult({ href: "http://localhost:8000/?github_oauth=<b>" }, h)).toBeNull();
    expect(h.replaceState).toHaveBeenCalledWith({ keep: 1 }, "", "/");
    expect(peekGitHubOAuthResult()).toBeNull();
  });
});

describe("the notice the page opens with", () => {
  it.each([pt, en])("says the outcome, or nothing", (t) => {
    const g = t.github;
    expect(gitHubOAuthNotice("connected", g)).toBe(g.oauthConnected);
    expect(gitHubOAuthNotice("denied", g)).toBe(g.oauthDenied);
    expect(gitHubOAuthNotice("failed", g)).toBe(g.oauthFailed);
    expect(gitHubOAuthNotice(null, g)).toBeNull();
    expect(new Set([g.oauthConnected, g.oauthDenied, g.oauthFailed]).size).toBe(3);
  });

  it("is said once: after the clear a reload's page has nothing to say", () => {
    const h = history();
    captureGitHubOAuthResult({ href: "http://localhost:8000/?github_oauth=connected" }, h);
    expect(gitHubOAuthNotice(peekGitHubOAuthResult(), pt.github)).toBe(pt.github.oauthConnected);
    clearGitHubOAuthResult();
    // The address no longer has the flag: a reload reads the cleaned one.
    const [, , cleaned] = h.replaceState.mock.calls[0] as [unknown, string, string];
    expect(captureGitHubOAuthResult({ href: `http://localhost:8000${cleaned}` }, history())).toBeNull();
    expect(gitHubOAuthNotice(peekGitHubOAuthResult(), pt.github)).toBeNull();
  });

  it("says the success in the words the lead asked for", () => {
    expect(pt.github.oauthConnected).toBe("GitHub conectado: publicar um run usa essa conta.");
    expect(en.github.oauthConnected).toContain("GitHub connected");
  });
});

describe("the button sends the browser only to GitHub's authorization page", () => {
  it.each([
    [AUTHORIZE, true],
    ["https://github.com/login/oauth/authorize", true],
    ["http://github.com/login/oauth/authorize?x=1", false],
    ["https://github.com.evil.example/login/oauth/authorize", false],
    ["https://evil@github.com/login/oauth/authorize", false],
    ["https://github.com/login/oauth/access_token", false],
    ["javascript:alert(1)", false],
    ["/login/oauth/authorize", false],
  ])("%s -> %s", (url, ok) => {
    expect(isGitHubAuthorizeUrl(url)).toBe(ok);
  });

  it("goes where the server says when it is GitHub", async () => {
    const go = vi.fn();
    await openGitHubAuthorization(async () => ({ url: AUTHORIZE }), go);
    expect(go).toHaveBeenCalledWith(AUTHORIZE);
  });

  it("stays when the server names another page", async () => {
    const go = vi.fn();
    await expect(
      openGitHubAuthorization(async () => ({ url: "https://evil.example/" }), go),
    ).rejects.toThrow();
    expect(go).not.toHaveBeenCalled();
  });

  it("stays when the server refuses", async () => {
    const go = vi.fn();
    await expect(
      openGitHubAuthorization(() => Promise.reject(new Error("off")), go),
    ).rejects.toThrow("off");
    expect(go).not.toHaveBeenCalled();
  });
});
