/** The Colab account as the device selector shows it (ADR-211). */

import { useEffect, useState } from "react";

import {
  fetchColabSessions,
  fetchColabStatus,
  fetchColabUsage,
  type ColabSessionInfo,
  type ColabStatus,
  type ColabUsage,
} from "../api/client";
import { colabStage, type ColabStage } from "../lib/colab-connection";
import { colabAccountFacts, type ColabAccountFacts } from "../lib/colab-target";
import { ROUTE, getSession, routeOpen } from "../lib/session";

export interface ColabAccount {
  stage: ColabStage | null;
  email: string | null;
  facts: ColabAccountFacts;
}

/**
 * The account as the selector shows it. The status is asked when the selector
 * mounts and each time its menu opens (a login made in the header since then
 * shows up); the balance and the runtimes only once connected, and only while
 * they are on screen.
 */
export function useColabAccount(open: boolean, active: boolean): ColabAccount {
  const [status, setStatus] = useState<ColabStatus | null>(null);
  const [usage, setUsage] = useState<ColabUsage | null>(null);
  const [sessions, setSessions] = useState<ColabSessionInfo[] | null>(null);

  useEffect(() => {
    if (!routeOpen(getSession(), ROUTE.colab)) return;
    let alive = true;
    fetchColabStatus()
      .then((s) => alive && setStatus(s))
      .catch(() => alive && setStatus(null));
    return () => {
      alive = false;
    };
  }, [open]);

  const stage = status ? colabStage(status) : null;
  const shown = stage === "connected" && (open || active);

  useEffect(() => {
    if (!shown) return;
    let alive = true;
    fetchColabUsage()
      .then((u) => alive && setUsage(u))
      .catch(() => alive && setUsage(null));
    fetchColabSessions()
      .then((s) => alive && setSessions(s))
      .catch(() => alive && setSessions(null));
    return () => {
      alive = false;
    };
  }, [shown, open]);

  return {
    stage,
    email: stage === "connected" ? (status?.account ?? null) : null,
    facts: colabAccountFacts(stage === "connected" ? usage : null, stage === "connected" ? sessions : null),
  };
}
