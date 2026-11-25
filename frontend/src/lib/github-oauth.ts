/** "Conectar com GitHub": the way back from GitHub (ADR-250).
 *
 * The server's callback sends the browser to the page with
 * `?github_oauth=connected` or `?github_oauth=failed`, never the code nor the
 * token. The flag is read once, when the page loads, and taken out of the
 * address so a reload does not show it again; the GitHub panel shows it the
 * next time it is drawn.
 */

export type GitHubOAuthResult = "connected" | "failed";

const PARAM = "github_oauth";
const AUTHORIZE = "https://github.com/login/oauth/authorize";

let pending: GitHubOAuthResult | null = null;

/** Read `?github_oauth=` and remove it from the address. */
export function captureGitHubOAuthResult(
  location: Pick<Location, "href">,
  history: Pick<History, "replaceState" | "state">,
): GitHubOAuthResult | null {
  const url = new URL(location.href);
  const value = url.searchParams.get(PARAM);
  if (value === null) return null;
  url.searchParams.delete(PARAM);
  history.replaceState(history.state, "", `${url.pathname}${url.search}${url.hash}`);
  pending = value === "connected" || value === "failed" ? value : null;
  return pending;
}

/** The outcome waiting to be shown, if any. */
export function peekGitHubOAuthResult(): GitHubOAuthResult | null {
  return pending;
}

/** The outcome was shown: forget it. */
export function clearGitHubOAuthResult(): void {
  pending = null;
}

/** True only for GitHub's own authorization page, the one place the button may
 *  send the browser to. */
export function isGitHubAuthorizeUrl(raw: string): boolean {
  try {
    const url = new URL(raw);
    return `${url.origin}${url.pathname}` === AUTHORIZE && url.username === "" && url.password === "";
  } catch {
    return false;
  }
}

/** Ask the server for a connection request and send the browser to GitHub.
 *  Anything but GitHub's authorization page is refused, without leaving. */
export async function openGitHubAuthorization(
  start: () => Promise<{ url: string }>,
  go: (url: string) => void,
): Promise<void> {
  const { url } = await start();
  if (!isGitHubAuthorizeUrl(url)) throw new Error();
  go(url);
}
