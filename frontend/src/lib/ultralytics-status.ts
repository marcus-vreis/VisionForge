import { useEffect, useState } from "react";
import { fetchSystemInfo, type SystemInfo } from "../api/client";

/** Whether the optional `detection` extra (ultralytics, AGPL-3.0) is installed,
 *  and the line that installs it (ADR-195). */
export type UltralyticsStatus = Pick<
  SystemInfo,
  "ultralytics_installed" | "detection_install_command"
>;

/** The install line to show under the backend choice, or null: only for the
 *  Ultralytics backend, and only once the server said it is absent. */
export function missingUltralyticsCommand(
  backend: string,
  status: UltralyticsStatus | null,
): string | null {
  if (backend !== "ultralytics" || status === null || status.ultralytics_installed) {
    return null;
  }
  return status.detection_install_command;
}

/** Asks the server once; null until it answers (or if it cannot). */
export function useUltralyticsStatus(): UltralyticsStatus | null {
  const [status, setStatus] = useState<UltralyticsStatus | null>(null);
  useEffect(() => {
    let alive = true;
    fetchSystemInfo()
      .then((info) => alive && setStatus(info))
      .catch(() => {
        /* Without an answer, say nothing: the submit's 422 still explains it. */
      });
    return () => {
      alive = false;
    };
  }, []);
  return status;
}
