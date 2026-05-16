import type { RunStatus } from "../types/run";
import { Alert, AlertDescription } from "./ui/alert";

interface ExperimentRunnerProps {
  status: RunStatus;
  error: string | null;
}

export function ExperimentRunner({ status, error }: ExperimentRunnerProps) {
  if (error) {
    return (
      <Alert variant="destructive">
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (status.status === "running") {
    return (
      <div className="flex items-center gap-3 rounded-md border border-border/50 p-4">
        <div className="h-4 w-4 rounded-full border-2 border-primary border-t-transparent animate-spin" />
        <div>
          <p className="text-sm font-medium">Training in progress...</p>
          <p className="text-xs text-muted-foreground">
            Run ID: {status.run_id}
          </p>
        </div>
      </div>
    );
  }

  return null;
}
