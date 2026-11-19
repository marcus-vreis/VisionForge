import { useT } from "../i18n/useT";
import { useSession } from "../lib/session";

/** Under a dataset field, on a shared server: the folders a dataset may live in.
 *
 * A server has no folder dialog to open (the dialog would open on the server's
 * own screen), so the "browse" button is gone there and the path is typed. This
 * says which paths the server accepts: the person's own folder first, then the
 * ones the operator shares. Nothing on a local install.
 */
export function ServerFoldersNote() {
  const t = useT();
  const session = useSession();
  if (!session.server_mode || session.dataset_dirs.length === 0) return null;
  return (
    <div
      role="note"
      aria-label={t.server.foldersLabel}
      style={{
        gridColumn: "1 / -1",
        padding: "10px 14px",
        border: "1px solid var(--vf-panel-stroke)",
        borderRadius: 10,
        fontSize: 13,
        lineHeight: 1.5,
        color: "var(--vf-text-dim)",
      }}
    >
      <div>{t.server.foldersTitle}</div>
      <ul style={{ margin: "6px 0", paddingLeft: 18 }}>
        {session.dataset_dirs.map((dir, index) => (
          <li key={dir}>
            <code
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                color: "var(--vf-text)",
                overflowWrap: "anywhere",
                userSelect: "all",
              }}
            >
              {dir}
            </code>{" "}
            ({index === 0 ? t.server.folderOwn : t.server.folderShared})
          </li>
        ))}
      </ul>
      <div>{t.server.foldersHint}</div>
    </div>
  );
}
