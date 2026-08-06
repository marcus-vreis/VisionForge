/** Detect a server process that predates the SPA it is serving.
 *
 * FastAPI serves `static/` from disk on every request, so `npm run build`
 * reaches the browser at once. Python modules do not work that way: they are
 * imported once, when the process starts. A `visionforge gui` left running
 * across a rebuild therefore serves **new JavaScript from old Python** — the
 * page asks for fields the stale routes never send, every guard on those fields
 * is false, and the feature quietly does nothing at all.
 *
 * That failure looks exactly like a broken build, which is the expensive part:
 * rebuilding and hard-reloading both "fail" to fix it, because neither is the
 * problem. The server records which bundle it booted against; the page knows its
 * own. If they differ, the answer is to restart the server.
 */

export interface HealthResponse {
  version: string;
  spa_bundle: string;
}

/** The filename of the bundle this code is running from, or "" if unknowable. */
export function currentBundleName(moduleUrl: string): string {
  const match = /\/(index-[A-Za-z0-9_-]+\.js)(?:[?#]|$)/.exec(moduleUrl);
  return match ? match[1] : "";
}

/** Whether the server is running code older than this page.
 *
 * Unknowable cases return false rather than warning: a dev server run from Vite
 * has no fingerprinted bundle, and an unbuilt checkout has no name to report.
 * Crying wolf there would train the researcher to ignore the banner.
 */
export function isServerStale(health: HealthResponse | null, bundle: string): boolean {
  if (!health || !health.spa_bundle || !bundle) return false;
  return health.spa_bundle !== bundle;
}

/** Ask the server whether it predates this page.
 *
 * A server older than the health endpoint itself answers `/api/health` with the
 * SPA's index.html, because the unknown route falls through to the catch-all.
 * That is not a failure to detect staleness — it *is* staleness, and the most
 * important case to catch, since it is the upgrade that introduces this check.
 */
export async function detectStaleServer(
  fetchFn: typeof fetch,
  moduleUrl: string,
): Promise<boolean> {
  const bundle = currentBundleName(moduleUrl);
  if (!bundle) return false;
  try {
    const resp = await fetchFn("/api/health");
    if (!resp.ok) return false;
    const body: unknown = await resp.json();
    const health = body as HealthResponse;
    return isServerStale(health, bundle);
  } catch {
    // Non-JSON from a 200 means the SPA fallback answered: no such route there.
    return true;
  }
}

export const STALE_SERVER_MESSAGE =
  "O servidor está rodando código Python mais antigo que esta página. " +
  "Reinicie o `visionforge gui` — recarregar o navegador não resolve.";
