import { artifactUrl } from "../api/client";
import type { RunResult } from "../types/run";
import { Badge } from "./ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Separator } from "./ui/separator";

interface ResultsViewProps {
  result: RunResult;
}

/** Format metric values for display. */
function formatMetric(value: unknown): string {
  if (value === null || value === undefined) return "N/A";
  if (typeof value === "number") {
    return value % 1 === 0 ? String(value) : value.toFixed(4);
  }
  return String(value);
}

/** Human-readable metric labels. */
const METRIC_LABELS: Record<string, string> = {
  best_val_loss: "Best Val Loss",
  best_epoch: "Best Epoch",
  total_epochs: "Total Epochs",
  test_accuracy: "Test Accuracy",
  test_f1: "Test F1",
  test_precision: "Test Precision",
  test_recall: "Test Recall",
  test_auc_roc: "Test AUC-ROC",
};

export function ResultsView({ result }: ResultsViewProps) {
  const graphics = result.artifacts?.graphics ?? [];

  return (
    <div className="grid gap-4">
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <CardTitle className="text-base">Experiment Results</CardTitle>
            <Badge variant="secondary" className="text-xs">
              {result.run_id}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
            {Object.entries(result.metrics).map(([key, value]) => (
              <div
                key={key}
                className="rounded-md border border-border/50 p-3 text-center"
              >
                <p className="text-xs text-muted-foreground mb-1">
                  {METRIC_LABELS[key] ?? key}
                </p>
                <p className="text-lg font-semibold tabular-nums">
                  {formatMetric(value)}
                </p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {graphics.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Plots</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {graphics.map((path, idx) => (
                <div key={idx} className="rounded-md border border-border/50 overflow-hidden">
                  <img
                    src={artifactUrl(path)}
                    alt={`Plot ${idx + 1}`}
                    className="w-full h-auto"
                  />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {result.report &&
        Object.keys(result.report).length > 0 && (
          <>
            <Separator />
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Report Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="text-xs bg-muted p-3 rounded-md overflow-x-auto">
                  {JSON.stringify(result.report, null, 2)}
                </pre>
              </CardContent>
            </Card>
          </>
        )}
    </div>
  );
}
