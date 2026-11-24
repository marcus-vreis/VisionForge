import { useT } from "../i18n/useT";
import { useSession } from "./session";

/** The line under a derived dataset's name. On a local install the result is
 *  written next to its source (`local`); on a shared server it goes into the
 *  person's own folder, copied when the source is a shared one (ADR-220). */
export function useDestHint(local: string): string {
  const t = useT();
  return useSession().server_mode ? t.server.destHint : local;
}
