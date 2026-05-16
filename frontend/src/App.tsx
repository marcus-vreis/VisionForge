import { ConfigForm } from "./components/ConfigForm";
import { ExperimentRunner } from "./components/ExperimentRunner";
import { ResultsView } from "./components/ResultsView";
import { useExperiment } from "./hooks/useExperiment";
import { Button } from "./components/ui/button";
import { Separator } from "./components/ui/separator";

export default function App() {
  const { status, result, error, validationErrors, submit, reset } =
    useExperiment();

  const isRunning = status.status === "running";
  const hasResult = result !== null;

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border/50 bg-card">
        <div className="mx-auto max-w-5xl px-4 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">
              VisionForge
            </h1>
            <p className="text-xs text-muted-foreground">
              Computer Vision Experimentation Platform
            </p>
          </div>
          {(hasResult || error) && (
            <Button variant="outline" size="sm" onClick={reset}>
              New Experiment
            </Button>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-6">
        {hasResult ? (
          <ResultsView result={result} />
        ) : (
          <div className="grid gap-4">
            <ExperimentRunner status={status} error={error} />

            {!isRunning && (
              <>
                {error && <Separator />}
                <ConfigForm
                  onSubmit={submit}
                  disabled={isRunning}
                  validationErrors={validationErrors}
                />
              </>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
