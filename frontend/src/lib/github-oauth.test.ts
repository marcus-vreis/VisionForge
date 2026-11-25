import { afterEach, describe, expect, it, vi } from "vitest";
import {
  captureGitHubOAuthResult,
  clearGitHubOAuthResult,
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
  it.each(["connected", "failed"] as const)("keeps %s and cleans the address", (result) => {
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
