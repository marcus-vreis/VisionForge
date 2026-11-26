import { useT } from "../i18n/useT";
import { useSession } from "../lib/session";

/** Beside a token field, on a shared server: how the server keeps what is saved.
 *
 * The operator decides it with a key in the server's environment (ADR-253); the
 * person saving a token cannot see that from the page otherwise. Nothing on a
 * local install or against a server that does not say.
 */
export function TokenStorageNote() {
  const t = useT();
  const session = useSession();
  if (!session.server_mode || session.tokens_encrypted === undefined) return null;
  return (
    <p role="note" style={{ fontSize: 12, lineHeight: 1.5, color: "var(--vf-text-dim)" }}>
      {session.tokens_encrypted ? t.server.tokensEncrypted : t.server.tokensPlain}
    </p>
  );
}
