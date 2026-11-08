import { expect, test } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { enter } from "./helpers";
import { pt } from "../src/i18n/pt";

const work = path.join(path.dirname(fileURLToPath(import.meta.url)), ".work");

test("Hub preview requires an explicit upload and keeps credentials out of export", async ({ page, request }) => {
  const runDir = path.join(work, "outputs", "models", "HubE2E", "20261108_120000_000001");
  mkdirSync(runDir, { recursive: true });
  writeFileSync(path.join(runDir, "run.json"), JSON.stringify({
    id: "HubE2E_20261108_120000_000001", experiment: "HubE2E", timestamp: "2026-11-08T12:00:00",
    status: "completed", run_dir: runDir, config: { name: "HubE2E", model: { name: "resnet18" } },
    metrics: { total_epochs: 1, best_val_accuracy: 0.8 }, history: [], artifacts: {}, tests: [],
  }));
  let configured = false;
  const uploads: unknown[] = [];
  const tokens: unknown[] = [];
  await page.route("**/api/hub/token", async (route) => {
    const req = route.request();
    if (req.method() === "PUT") { tokens.push(req.postDataJSON()); configured = true; }
    if (req.method() === "DELETE") configured = false;
    await route.fulfill({ json: { configured, shared: true } });
  });
  await page.route("**/api/hub/preview?*", (route) => route.fulfill({ json: {
    files: ["README.md", "run.json", "model.pth"], digest: "reviewed-digest", card: "# Reviewed model card",
  } }));
  await page.route("**/api/hub/export", async (route) => {
    uploads.push(route.request().postDataJSON());
    await route.fulfill({ json: { url: "https://huggingface.co/test/reviewed" } });
  });
  await enter(page);
  await page.locator("[data-tour='history']").click();
  await page.getByTestId("history-overlay").getByText("HubE2E", { exact: true }).click();
  await page.getByRole("button", { name: pt.hub.open, exact: true }).click();
  const panel = page.getByRole("region", { name: pt.hub.title });
  await expect(panel.getByText(pt.hub.shared, { exact: true })).toBeVisible();
  await expect(panel.getByRole("list")).toHaveText("README.mdrun.jsonmodel.pth");
  await panel.getByText(pt.hub.card, { exact: true }).click();
  await expect(panel.getByText("# Reviewed model card", { exact: true })).toBeVisible();
  await expect(panel.getByText(pt.hub.checkpointWarning, { exact: true })).toBeVisible();
  await panel.getByLabel(pt.hub.repo, { exact: true }).fill("test/reviewed");
  const send = panel.getByRole("button", { name: pt.hub.send, exact: true });
  await expect(send).toBeDisabled();
  await panel.getByLabel(pt.hub.token, { exact: true }).fill("hf_placeholder_not_a_real_secret");
  await panel.getByRole("button", { name: pt.hub.saveToken, exact: true }).click();
  await expect(panel.getByLabel(pt.hub.token, { exact: true })).toHaveValue("");
  await expect(send).toBeEnabled();
  expect(uploads).toEqual([]);
  expect(tokens).toEqual([{ token: "hf_placeholder_not_a_real_secret" }]);
  await send.click();
  await expect(panel.getByRole("link", { name: pt.hub.success })).toHaveAttribute("href", "https://huggingface.co/test/reviewed");
  const runs = await (await request.get("/api/runs")).json();
  expect(uploads).toEqual([{ run_id: runs[0].run_id, repo_id: "test/reviewed", digest: "reviewed-digest" }]);
  await panel.getByRole("button", { name: pt.hub.removeToken, exact: true }).click();
  await expect(send).toBeDisabled();
});
