import { defineConfig, devices } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * End-to-end suite: the real GUI against a real local server, training tiny
 * synthetic datasets on the CPU.
 *
 * It tests the *built* SPA, the bundle `npm run build` writes to
 * src/visionforge/gui/static and the server serves — not the Vite dev server.
 * Rebuild before running it locally, or it exercises a stale bundle.
 */

const here = path.dirname(fileURLToPath(import.meta.url));
const CI = !!process.env.CI;
const PORT = Number(process.env.VF_E2E_PORT ?? 8765);

// The `visionforge` executable to start. CI installs it on the PATH; locally the
// project's own venv is the default.
const executable =
  process.env.VF_E2E_EXECUTABLE ??
  (CI
    ? "visionforge"
    : process.platform === "win32"
      ? path.resolve(here, "../.venv/Scripts/visionforge.exe")
      : path.resolve(here, "../.venv/bin/visionforge"));

export default defineConfig({
  testDir: "./e2e",
  // Not `*.spec.ts`: vitest collects that pattern and would try to run these.
  testMatch: "**/*.e2e.ts",
  // One server and one training slot: tests that train would queue behind each other.
  fullyParallel: false,
  workers: 1,
  // No retries: the runs of a failed attempt stay in the server's work folder and
  // would be found by the retry, which proves nothing.
  retries: 0,
  forbidOnly: CI,
  timeout: 180_000,
  expect: { timeout: 15_000 },
  reporter: CI ? [["list"], ["html", { open: "never" }]] : [["list"]],
  use: {
    launchOptions: { args: ["--disable-gpu"] },
    baseURL: `http://127.0.0.1:${PORT}`,
    locale: "pt-BR",
    actionTimeout: 15_000,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "chromium",
      // Locally the Chrome already installed on the machine; CI installs Playwright's chromium.
      use: CI ? { ...devices["Desktop Chrome"] } : { ...devices["Desktop Chrome"], channel: "chrome" },
    },
  ],
  webServer: {
    command: "node e2e/serve.mjs",
    url: `http://127.0.0.1:${PORT}/api/health`,
    // A fresh work folder per run: never attach to a server that has someone's runs.
    reuseExistingServer: false,
    timeout: 120_000,
    stdout: "pipe",
    stderr: "pipe",
    env: {
      VF_E2E_EXECUTABLE: executable,
      VF_E2E_PORT: String(PORT),
    },
  },
});
