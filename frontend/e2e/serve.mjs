// Starts `visionforge gui` for the end-to-end suite from a fresh, throwaway work
// folder, so the runs and sample datasets the suite creates never land in the
// user's outputs/ or datasets/ (the server resolves both against its cwd).
import { spawn } from "node:child_process";
import { mkdirSync, rmSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
// VF_E2E_SERVER_MODE starts the shared-server mode (ADR-212) in a folder of its
// own. The suite's browser plays the login proxy: this machine is the trusted
// address, and each test names its account in the proxy's header.
const serverMode = process.env.VF_E2E_SERVER_MODE === "1";
const work = path.join(here, serverMode ? ".work-server" : ".work");
const exe = process.env.VF_E2E_EXECUTABLE;
const port = process.env.VF_E2E_PORT;
if (!exe || !port) {
  console.error("serve.mjs: VF_E2E_EXECUTABLE and VF_E2E_PORT are set by playwright.config.ts");
  process.exit(2);
}

rmSync(work, { recursive: true, force: true });
mkdirSync(work, { recursive: true });
const shared = path.join(work, "shared");
if (serverMode) mkdirSync(shared);

// An explicit interpreter also supports worktrees that share the original venv.
const python = process.env.VF_E2E_PYTHON;
const args = ["gui", "--port", port, "--no-browser"];
if (serverMode) args.push("--server", "--trusted-proxy", "127.0.0.1", "--datasets-dir", shared);
const child = spawn(python ?? exe, python ? ["-m", "visionforge", ...args] : args, {
  cwd: work,
  stdio: "inherit",
  env: {
    ...process.env,
    // The suite trains on the CPU even when the UI would offer a GPU: the
    // machine's GPU may be busy with a real training. "-1", not "": on Windows
    // an empty value hides nothing, and the local suite then saw the GPU.
    CUDA_VISIBLE_DEVICES: "-1",
    OMP_NUM_THREADS: "1",
    MKL_NUM_THREADS: "1",
    OPENBLAS_NUM_THREADS: "1",
    VISIONFORGE_NO_BROWSER: "1",
  },
});

const stop = () => child.kill();
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
child.on("exit", (code) => process.exit(code ?? 1));
