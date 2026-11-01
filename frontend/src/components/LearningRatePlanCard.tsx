import { useEffect, useState } from "react";
import {
  fetchLearningRatePlan,
  type LearningRatePlanRequest,
  type TrialLearningRate,
} from "../api/client";
import { useT } from "../i18n/useT";
import { collapseWarning } from "../lib/lr-plan";

// The form changes on each key press and every step stays mounted.
const REFRESH_DELAY_MS = 250;

/** The rate each trial will really train at, and which combinations were
 *  measured to fail (ADR-158). Random search has no such list: its rates are
 *  sampled when it runs. */
export function LearningRatePlanCard({
  request,
}: {
  request: LearningRatePlanRequest | null;
}) {
  const t = useT();
  const words = t.stepShell.review.lrPlan;
  const [trials, setTrials] = useState<TrialLearningRate[] | null>(null);
  const [failed, setFailed] = useState(false);
  const key = request ? JSON.stringify(request) : null;

  useEffect(() => {
    if (!key) return;
    let current = true;
    const timer = setTimeout(() => {
      fetchLearningRatePlan(JSON.parse(key) as LearningRatePlanRequest)
        .then((plan) => {
          if (!current) return;
          setTrials(plan.trials);
          setFailed(false);
        })
        .catch(() => {
          if (!current) return;
          // A stale table would state rates that are no longer the ones used.
          setTrials(null);
          setFailed(true);
        });
    }, REFRESH_DELAY_MS);
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [key]);

  if (failed) return <div className="vf-review-lr-note">{words.unavailable}</div>;
  if (!trials || trials.length === 0) return null;

  const warnings = trials.flatMap((trial) => collapseWarning(t, trial) ?? []);
  const hasMultiplier = trials.some((trial) => trial.multiplier !== null);

  return (
    <div className="vf-review-lr">
      <div className="vf-review-lr-title">{words.title}</div>
      <div style={{ overflowX: "auto" }}>
        <table>
          <thead>
            <tr>
              <th>{words.model}</th>
              <th>{words.optimizer}</th>
              {hasMultiplier && <th>{words.multiplier}</th>}
              <th>{words.effectiveRate}</th>
            </tr>
          </thead>
          <tbody>
            {trials.map((trial) => (
              <tr
                key={`${trial.architecture}/${trial.optimizer}/${trial.learning_rate}`}
                data-flagged={trial.collapse_prone ? "" : undefined}
              >
                <td>{trial.architecture}</td>
                <td>{trial.optimizer}</td>
                {hasMultiplier && <td>{`${trial.multiplier}×`}</td>}
                <td>{trial.learning_rate}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {warnings.length > 0 && (
        <ul role="alert">
          {warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
