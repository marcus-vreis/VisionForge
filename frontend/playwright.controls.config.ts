import { defineConfig } from "@playwright/test";
import config from "./playwright.config";

/** Exercise shared controls in a real browser without a backend or training. */
export default defineConfig({
  ...config,
  testMatch: "**/controls.e2e.ts",
  // Only the plain browser project: the server-mode one brings its own tests.
  projects: config.projects?.filter((project) => project.name === "chromium"),
  webServer: undefined,
});
